//! Per-client connection for the Minecraft protocol.
//!
//! Each accepted TCP connection is represented by a [`Connection`] that
//! tracks the remote address, protocol state, and provides methods to
//! read/write raw packet frames.
//!
//! Supports optional encryption (AES-128-CFB8) and compression (zlib)
//! which are enabled during the login handshake.

use std::fmt;
use std::io;
use std::net::SocketAddr;

use bytes::{Bytes, BytesMut};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};

use crate::handle::HandleError;
use oxidized_codec::frame::{self, FrameError, MAX_PACKET_SIZE};
use oxidized_codec::packet::{Packet, PacketDecodeError};
use oxidized_codec::varint::{self, VarIntError};
use oxidized_compression::{CompressionError, CompressionState, Compressor, Decompressor};
use oxidized_crypto::{CipherState, DecryptCipher, EncryptCipher};

// ---------------------------------------------------------------------------
// ConnectionState
// ---------------------------------------------------------------------------

/// Protocol state of a Minecraft connection.
///
/// Connections start in [`Handshaking`](ConnectionState::Handshaking) and
/// transition based on the client's intention packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Initial state — waiting for the handshake packet.
    Handshaking,
    /// Server list ping / status query.
    Status,
    /// Authentication / login flow.
    Login,
    /// Configuration state (1.20.2+).
    Configuration,
    /// Main gameplay state.
    Play,
}

impl fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handshaking => write!(f, "Handshaking"),
            Self::Status => write!(f, "Status"),
            Self::Login => write!(f, "Login"),
            Self::Configuration => write!(f, "Configuration"),
            Self::Play => write!(f, "Play"),
        }
    }
}

// ---------------------------------------------------------------------------
// ConnectionStateMachine
// ---------------------------------------------------------------------------

/// Invalid state transition error.
///
/// Returned when [`ConnectionStateMachine::transition`] is called with a
/// `(from, to)` pair that doesn't match any valid Minecraft protocol
/// transition.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid state transition: {from} -> {to}")]
pub struct InvalidTransition {
    /// The state the connection was in.
    pub from: ConnectionState,
    /// The state the caller tried to move to.
    pub to: ConnectionState,
}

/// Testable protocol state machine, decoupled from network I/O.
///
/// Tracks the current [`ConnectionState`] and validates that transitions
/// follow the Minecraft protocol:
///
/// ```text
/// Handshaking → Status         (server list ping)
/// Handshaking → Login          (authentication / protocol mismatch)
/// Login       → Configuration  (login success)
/// Configuration → Play         (configuration finished)
/// ```
///
/// # Examples
///
/// ```
/// use oxidized_transport::connection::{ConnectionState, ConnectionStateMachine};
///
/// let mut sm = ConnectionStateMachine::new();
/// assert_eq!(sm.state(), ConnectionState::Handshaking);
///
/// sm.transition(ConnectionState::Login).unwrap();
/// assert_eq!(sm.state(), ConnectionState::Login);
///
/// sm.transition(ConnectionState::Configuration).unwrap();
/// assert_eq!(sm.state(), ConnectionState::Configuration);
///
/// sm.transition(ConnectionState::Play).unwrap();
/// assert_eq!(sm.state(), ConnectionState::Play);
/// ```
#[derive(Debug, Clone)]
pub struct ConnectionStateMachine {
    state: ConnectionState,
    /// Whether encryption has been enabled on this connection.
    is_encrypted: bool,
    /// Whether compression has been enabled on this connection.
    is_compressed: bool,
}

impl ConnectionStateMachine {
    /// Creates a new state machine in the [`Handshaking`](ConnectionState::Handshaking)
    /// state.
    pub fn new() -> Self {
        Self {
            state: ConnectionState::Handshaking,
            is_encrypted: false,
            is_compressed: false,
        }
    }

    /// Returns the current protocol state.
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Returns whether encryption is enabled.
    pub fn is_encrypted(&self) -> bool {
        self.is_encrypted
    }

    /// Returns whether compression is enabled.
    pub fn is_compressed(&self) -> bool {
        self.is_compressed
    }

    /// Records that encryption has been enabled.
    pub fn set_encrypted(&mut self) {
        self.is_encrypted = true;
    }

    /// Records that compression has been enabled.
    pub fn set_compressed(&mut self) {
        self.is_compressed = true;
    }

    /// Attempts to transition to the given state.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidTransition`] if the `(current, target)` pair is
    /// not a valid protocol transition.
    pub fn transition(&mut self, to: ConnectionState) -> Result<(), InvalidTransition> {
        if Self::is_valid_transition(self.state, to) {
            self.state = to;
            Ok(())
        } else {
            Err(InvalidTransition {
                from: self.state,
                to,
            })
        }
    }

    /// Returns `true` if `from → to` is a valid protocol transition.
    pub fn is_valid_transition(from: ConnectionState, to: ConnectionState) -> bool {
        matches!(
            (from, to),
            (ConnectionState::Handshaking, ConnectionState::Status)
                | (ConnectionState::Handshaking, ConnectionState::Login)
                | (ConnectionState::Login, ConnectionState::Configuration)
                | (ConnectionState::Configuration, ConnectionState::Play)
        )
    }
}

impl Default for ConnectionStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RawPacket
// ---------------------------------------------------------------------------

/// A raw (undecoded) packet: just the numeric ID and the body bytes.
#[derive(Debug, Clone)]
pub struct RawPacket {
    /// Packet ID (VarInt on the wire, decoded to i32).
    pub id: i32,
    /// Packet body bytes (everything after the packet ID).
    pub data: Bytes,
}

// ---------------------------------------------------------------------------
// ConnectionError
// ---------------------------------------------------------------------------

/// Errors that can occur on a [`Connection`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConnectionError {
    /// A frame-level error (bad length prefix, oversized packet, etc.).
    #[error("frame error: {0}")]
    Frame(#[from] FrameError),

    /// A VarInt decoding error in the packet ID.
    #[error("packet ID decode error: {0}")]
    VarInt(#[from] VarIntError),

    /// A compression/decompression error.
    #[error("compression error: {0}")]
    Compression(#[from] CompressionError),

    /// An I/O error on the underlying TCP stream.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// A protocol-level error when decoding a typed packet.
    #[error("protocol error: {0}")]
    Protocol(#[from] PacketDecodeError),

    /// The client exceeded the packet rate limit.
    #[error("rate limited: client exceeded {0} packets per tick window")]
    RateLimited(u32),

    /// The outbound channel is closed or full (writer task exited or slow client).
    #[error("outbound channel closed")]
    ChannelClosed(#[from] HandleError),
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

/// A single client connection.
///
/// Owns the split TCP stream halves and tracks protocol state.
/// Supports optional AES-128-CFB8 encryption and zlib compression
/// which are enabled during the login handshake.
pub struct Connection {
    reader: OwnedReadHalf,
    writer: OwnedWriteHalf,
    addr: SocketAddr,

    /// Current protocol state.
    pub state: ConnectionState,
    /// Protocol version reported by the client in the handshake.
    pub protocol_version: i32,

    /// AES-128-CFB8 cipher state (enabled after key exchange).
    cipher: Option<CipherState>,
    /// Zlib compression state (enabled after login compression packet).
    compression: Option<CompressionState>,
}

impl Connection {
    /// Creates a new connection from an accepted [`TcpStream`].
    ///
    /// Sets `TCP_NODELAY` for low-latency writes and splits
    /// the stream into independent read/write halves.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if `TCP_NODELAY` cannot be set.
    pub fn new(stream: TcpStream, addr: SocketAddr) -> io::Result<Self> {
        stream.set_nodelay(true)?;
        let (reader, writer) = stream.into_split();
        Ok(Self {
            reader,
            writer,
            addr,
            state: ConnectionState::Handshaking,
            protocol_version: 0,
            cipher: None,
            compression: None,
        })
    }

    /// Returns the remote socket address.
    pub fn remote_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Enables AES-128-CFB8 encryption on this connection.
    ///
    /// After calling this, all subsequent reads and writes will be
    /// encrypted/decrypted using the shared secret.
    pub fn enable_encryption(&mut self, shared_secret: &[u8; 16]) {
        self.cipher = Some(CipherState::new(shared_secret));
    }

    /// Returns whether encryption is enabled.
    pub fn is_encrypted(&self) -> bool {
        self.cipher.is_some()
    }

    /// Enables zlib compression on this connection with the given threshold.
    ///
    /// After calling this, packets at or above `threshold` bytes will be
    /// zlib-compressed. The frame format changes to include a `data_length`
    /// VarInt prefix.
    pub fn enable_compression(&mut self, threshold: usize) {
        self.compression = Some(CompressionState::new(threshold));
    }

    /// Returns whether compression is enabled.
    pub fn is_compressed(&self) -> bool {
        self.compression.is_some()
    }

    // -----------------------------------------------------------------------
    // Low-level encrypted I/O
    // -----------------------------------------------------------------------

    /// Reads exactly `n` bytes from the TCP stream, decrypting if needed.
    async fn read_bytes(&mut self, n: usize) -> Result<BytesMut, io::Error> {
        let mut buf = BytesMut::zeroed(n);
        self.reader.read_exact(&mut buf).await?;
        if let Some(ref mut cipher) = self.cipher {
            cipher.decrypt(&mut buf);
        }
        Ok(buf)
    }

    /// Reads a single byte from the TCP stream, decrypting if needed.
    async fn read_byte(&mut self) -> Result<u8, io::Error> {
        let mut byte = [0u8; 1];
        self.reader.read_exact(&mut byte).await?;
        if let Some(ref mut cipher) = self.cipher {
            cipher.decrypt(&mut byte);
        }
        Ok(byte[0])
    }

    /// Writes raw bytes to the TCP stream, encrypting if needed.
    async fn write_bytes(&mut self, data: &mut [u8]) -> Result<(), io::Error> {
        if let Some(ref mut cipher) = self.cipher {
            cipher.encrypt(data);
        }
        self.writer.write_all(data).await
    }

    // -----------------------------------------------------------------------
    // Frame reading (with encryption + compression)
    // -----------------------------------------------------------------------

    /// Reads a VarInt from the (possibly encrypted) stream.
    async fn read_varint(&mut self) -> Result<i32, ConnectionError> {
        let mut result: i32 = 0;
        for i in 0..varint::VARINT_MAX_BYTES {
            let byte = self.read_byte().await?;
            result |= ((byte & 0x7F) as i32) << (7 * i);
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err(ConnectionError::VarInt(VarIntError::TooLarge {
            max_bytes: varint::VARINT_MAX_BYTES,
        }))
    }

    /// Reads one raw packet from the connection.
    ///
    /// Handles the full pipeline: decrypt → read frame → decompress →
    /// extract packet ID.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure, malformed framing,
    /// oversized packets, or decompression errors.
    pub async fn read_raw_packet(&mut self) -> Result<RawPacket, ConnectionError> {
        // Step 1: Read frame (encrypted bytes are decrypted transparently)
        let frame_payload = if self.cipher.is_some() {
            // Encrypted path: read VarInt + payload through decrypt layer
            let length = self.read_varint().await?;
            let length = length as usize;
            if length == 0 {
                return Err(ConnectionError::Frame(FrameError::ZeroLength));
            }
            if length > MAX_PACKET_SIZE {
                return Err(ConnectionError::Frame(FrameError::PacketTooLarge {
                    size: length,
                    max: MAX_PACKET_SIZE,
                }));
            }
            let buf = self.read_bytes(length).await?;
            buf.freeze()
        } else {
            // Unencrypted path: use existing frame reader
            frame::read_frame(&mut self.reader, MAX_PACKET_SIZE).await?
        };

        // Step 2: Handle compression (if enabled)
        let packet_data = if let Some(ref mut compression) = self.compression {
            let mut buf = frame_payload;
            let data_length = varint::read_varint_buf(&mut buf)?;
            let decompressed = compression.decompress(data_length, &buf)?;
            Bytes::from(decompressed)
        } else {
            frame_payload
        };

        // Step 3: Parse packet ID
        let mut buf = packet_data;
        let id = varint::read_varint_buf(&mut buf)?;
        Ok(RawPacket { id, data: buf })
    }

    /// Sends a raw packet (ID + body) as a single frame.
    ///
    /// Handles the full pipeline: build payload → compress → frame →
    /// encrypt → write.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure or compression errors.
    pub async fn send_raw(&mut self, id: i32, data: &[u8]) -> Result<(), ConnectionError> {
        // Step 1: Build inner payload (packet_id + body)
        let mut inner = BytesMut::new();
        varint::write_varint_buf(id, &mut inner);
        inner.extend_from_slice(data);

        // Step 2: Handle compression (if enabled)
        let frame_content = if let Some(ref mut compression) = self.compression {
            let (data_length, payload) = compression.compress(&inner)?;
            let mut compressed_frame = BytesMut::new();
            varint::write_varint_buf(data_length, &mut compressed_frame);
            compressed_frame.extend_from_slice(&payload);
            compressed_frame
        } else {
            inner
        };

        // Step 3: Build frame (VarInt length prefix + content)
        let mut frame = BytesMut::new();
        varint::write_varint_buf(frame_content.len() as i32, &mut frame);
        frame.extend_from_slice(&frame_content);

        // Step 4: Encrypt (if enabled) and write
        let mut frame_bytes = frame.to_vec();
        self.write_bytes(&mut frame_bytes).await?;
        Ok(())
    }

    /// Flushes the write buffer, ensuring all data reaches the OS send buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure.
    pub async fn flush(&mut self) -> Result<(), ConnectionError> {
        self.writer.flush().await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Generic typed packet I/O
    // -----------------------------------------------------------------------

    /// Sends a typed packet (encodes, frames, and flushes).
    ///
    /// This is the high-level send API — it encodes the packet body via
    /// [`Packet::encode`], prepends the packet ID via [`send_raw`](Self::send_raw),
    /// and flushes the write buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure or compression errors.
    pub async fn send_packet<P: Packet>(&mut self, pkt: &P) -> Result<(), ConnectionError> {
        let body = pkt.encode();
        self.send_raw(P::PACKET_ID, &body).await?;
        self.flush().await
    }

    /// Decodes a [`RawPacket`] into a typed packet.
    ///
    /// The caller is responsible for checking that `raw.id` matches
    /// `P::PACKET_ID` — this method only decodes the body bytes.
    ///
    /// # Errors
    ///
    /// Returns [`PacketDecodeError`] if the body bytes cannot be decoded
    /// into the target packet type.
    pub fn decode_packet<P: Packet>(raw: &RawPacket) -> Result<P, PacketDecodeError> {
        P::decode(raw.data.clone())
    }

    /// Shuts down the write half of the connection.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure.
    pub async fn shutdown(&mut self) -> Result<(), ConnectionError> {
        self.writer.shutdown().await?;
        Ok(())
    }

    /// Consumes this connection and splits it into independent reader and
    /// writer halves.
    ///
    /// Each half owns its own cipher (decrypt/encrypt) and compression
    /// (decompress/compress) state. Call this at the Configuration → Play
    /// state transition to spawn the reader/writer task pair.
    ///
    /// The `Connection` is consumed and cannot be used after this call.
    pub fn into_split(self) -> (ConnectionReader, ConnectionWriter) {
        let (decrypt, encrypt) = match self.cipher {
            Some(cipher) => {
                let (d, e) = cipher.split();
                (Some(d), Some(e))
            },
            None => (None, None),
        };
        let (decompressor, compressor) = match self.compression {
            Some(comp) => {
                let (d, c) = comp.split();
                (Some(d), Some(c))
            },
            None => (None, None),
        };
        (
            ConnectionReader {
                reader: self.reader,
                addr: self.addr,
                decrypt,
                decompressor,
            },
            ConnectionWriter {
                writer: self.writer,
                addr: self.addr,
                encrypt,
                compressor,
                batch_buf: BytesMut::with_capacity(INITIAL_BATCH_BUF_CAPACITY),
            },
        )
    }
}

impl fmt::Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Connection")
            .field("addr", &self.addr)
            .field("state", &self.state)
            .field("protocol_version", &self.protocol_version)
            .field("encrypted", &self.cipher.is_some())
            .field("compressed", &self.compression.is_some())
            .finish()
    }
}

/// Initial capacity for the writer's batch buffer (64 KB).
const INITIAL_BATCH_BUF_CAPACITY: usize = 64 * 1024;

// ---------------------------------------------------------------------------
// ConnectionReader
// ---------------------------------------------------------------------------

/// Read half of a split connection.
///
/// Owns the TCP read half and the decryption/decompression state.
/// Produced by [`Connection::into_split`] and used by the reader task
/// to receive packets from the client.
pub struct ConnectionReader {
    reader: OwnedReadHalf,
    addr: SocketAddr,
    decrypt: Option<DecryptCipher>,
    decompressor: Option<Decompressor>,
}

impl ConnectionReader {
    /// Returns the remote socket address.
    pub fn remote_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Reads exactly `n` bytes, decrypting if needed.
    async fn read_bytes(&mut self, n: usize) -> Result<BytesMut, io::Error> {
        let mut buf = BytesMut::zeroed(n);
        self.reader.read_exact(&mut buf).await?;
        if let Some(ref mut cipher) = self.decrypt {
            cipher.decrypt(&mut buf);
        }
        Ok(buf)
    }

    /// Reads a single byte, decrypting if needed.
    async fn read_byte(&mut self) -> Result<u8, io::Error> {
        let mut byte = [0u8; 1];
        self.reader.read_exact(&mut byte).await?;
        if let Some(ref mut cipher) = self.decrypt {
            cipher.decrypt(&mut byte);
        }
        Ok(byte[0])
    }

    /// Reads a VarInt from the (possibly encrypted) stream.
    async fn read_varint(&mut self) -> Result<i32, ConnectionError> {
        let mut result: i32 = 0;
        for i in 0..varint::VARINT_MAX_BYTES {
            let byte = self.read_byte().await?;
            result |= ((byte & 0x7F) as i32) << (7 * i);
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err(ConnectionError::VarInt(VarIntError::TooLarge {
            max_bytes: varint::VARINT_MAX_BYTES,
        }))
    }

    /// Reads one raw packet from the connection.
    ///
    /// Handles the full pipeline: decrypt → read frame → decompress →
    /// extract packet ID.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure, malformed framing,
    /// oversized packets, or decompression errors.
    pub async fn read_raw_packet(&mut self) -> Result<RawPacket, ConnectionError> {
        let frame_payload = if self.decrypt.is_some() {
            let length = self.read_varint().await?;
            let length = length as usize;
            if length == 0 {
                return Err(ConnectionError::Frame(FrameError::ZeroLength));
            }
            if length > MAX_PACKET_SIZE {
                return Err(ConnectionError::Frame(FrameError::PacketTooLarge {
                    size: length,
                    max: MAX_PACKET_SIZE,
                }));
            }
            let buf = self.read_bytes(length).await?;
            buf.freeze()
        } else {
            frame::read_frame(&mut self.reader, MAX_PACKET_SIZE).await?
        };

        let packet_data = if let Some(ref mut decompressor) = self.decompressor {
            let mut buf = frame_payload;
            let data_length = varint::read_varint_buf(&mut buf)?;
            let decompressed = decompressor.decompress(data_length, &buf)?;
            Bytes::from(decompressed)
        } else {
            frame_payload
        };

        let mut buf = packet_data;
        let id = varint::read_varint_buf(&mut buf)?;
        Ok(RawPacket { id, data: buf })
    }
}

impl fmt::Debug for ConnectionReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConnectionReader")
            .field("addr", &self.addr)
            .field("encrypted", &self.decrypt.is_some())
            .field("compressed", &self.decompressor.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ConnectionWriter
// ---------------------------------------------------------------------------

/// Write half of a split connection.
///
/// Owns the TCP write half and the encryption/compression state.
/// Produced by [`Connection::into_split`] and used by the writer task
/// to send packets to the client.
///
/// The `batch_buf` is a pre-allocated buffer used by the writer task
/// (R4.3) to batch multiple packets into a single write syscall.
pub struct ConnectionWriter {
    writer: OwnedWriteHalf,
    addr: SocketAddr,
    encrypt: Option<EncryptCipher>,
    compressor: Option<Compressor>,
    /// Pre-allocated buffer for batch encoding (used by writer task).
    batch_buf: BytesMut,
}

impl ConnectionWriter {
    /// Returns the remote socket address.
    pub fn remote_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Writes raw bytes, encrypting if needed.
    async fn write_bytes(&mut self, data: &mut [u8]) -> Result<(), io::Error> {
        if let Some(ref mut cipher) = self.encrypt {
            cipher.encrypt(data);
        }
        self.writer.write_all(data).await
    }

    /// Sends a raw packet (ID + body) as a single frame.
    ///
    /// Handles the full pipeline: build payload → compress → frame →
    /// encrypt → write.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure or compression errors.
    pub async fn send_raw(&mut self, id: i32, data: &[u8]) -> Result<(), ConnectionError> {
        let mut inner = BytesMut::new();
        varint::write_varint_buf(id, &mut inner);
        inner.extend_from_slice(data);

        let frame_content = if let Some(ref mut compressor) = self.compressor {
            let (data_length, payload) = compressor.compress(&inner)?;
            let mut compressed_frame = BytesMut::new();
            varint::write_varint_buf(data_length, &mut compressed_frame);
            compressed_frame.extend_from_slice(&payload);
            compressed_frame
        } else {
            inner
        };

        let mut frame = BytesMut::new();
        varint::write_varint_buf(frame_content.len() as i32, &mut frame);
        frame.extend_from_slice(&frame_content);

        let mut frame_bytes = frame.to_vec();
        self.write_bytes(&mut frame_bytes).await?;
        Ok(())
    }

    /// Sends a typed packet (encodes, frames, and flushes).
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure or compression errors.
    pub async fn send_packet<P: Packet>(&mut self, pkt: &P) -> Result<(), ConnectionError> {
        let body = pkt.encode();
        self.send_raw(P::PACKET_ID, &body).await?;
        self.flush().await
    }

    /// Flushes the write buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure.
    pub async fn flush(&mut self) -> Result<(), ConnectionError> {
        self.writer.flush().await?;
        Ok(())
    }

    /// Shuts down the write half.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure.
    pub async fn shutdown(&mut self) -> Result<(), ConnectionError> {
        self.writer.shutdown().await?;
        Ok(())
    }

    /// Returns a mutable reference to the batch buffer.
    ///
    /// Used by the writer task (R4.3) for batch-encoding packets before
    /// a single flush.
    pub fn batch_buf_mut(&mut self) -> &mut BytesMut {
        &mut self.batch_buf
    }

    /// Returns the current batch buffer length.
    pub fn batch_buf_len(&self) -> usize {
        self.batch_buf.len()
    }

    // -----------------------------------------------------------------------
    // Batch encoding (R4.3 — writer task)
    // -----------------------------------------------------------------------

    /// Encodes a single packet into the batch buffer without writing.
    ///
    /// Builds the complete frame (packet ID + body → compress → length
    /// prefix) and appends it to the internal batch buffer. Call
    /// [`flush_batch`](Self::flush_batch) to encrypt and write the
    /// accumulated batch to the network.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on compression errors.
    pub fn encode_to_batch(&mut self, id: i32, data: &[u8]) -> Result<(), ConnectionError> {
        // Step 1: Build inner payload (packet_id + body)
        let mut inner = BytesMut::new();
        varint::write_varint_buf(id, &mut inner);
        inner.extend_from_slice(data);

        // Step 2: Handle compression (if enabled)
        let frame_content = if let Some(ref mut compressor) = self.compressor {
            let (data_length, payload) = compressor.compress(&inner)?;
            let mut compressed_frame = BytesMut::new();
            varint::write_varint_buf(data_length, &mut compressed_frame);
            compressed_frame.extend_from_slice(&payload);
            compressed_frame
        } else {
            inner
        };

        // Step 3: Append frame (VarInt length prefix + content) to batch
        varint::write_varint_buf(frame_content.len() as i32, &mut self.batch_buf);
        self.batch_buf.extend_from_slice(&frame_content);

        Ok(())
    }

    /// Encrypts and writes the entire batch buffer, then clears it.
    ///
    /// Encrypts all accumulated frames in-place (if encryption is
    /// enabled), writes the batch to the TCP stream in a single
    /// `write_all` call, flushes, and clears the buffer for reuse.
    ///
    /// If the batch buffer is empty, this is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectionError`] on I/O failure.
    pub async fn flush_batch(&mut self) -> Result<(), ConnectionError> {
        if self.batch_buf.is_empty() {
            return Ok(());
        }

        // Encrypt entire batch in-place
        if let Some(ref mut cipher) = self.encrypt {
            cipher.encrypt(&mut self.batch_buf);
        }

        // Single write for the entire batch
        self.writer.write_all(&self.batch_buf).await?;
        self.writer.flush().await?;
        self.batch_buf.clear();
        Ok(())
    }
}

impl fmt::Debug for ConnectionWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConnectionWriter")
            .field("addr", &self.addr)
            .field("encrypted", &self.encrypt.is_some())
            .field("compressed", &self.compressor.is_some())
            .field("batch_buf_len", &self.batch_buf.len())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    /// Helper: creates a connected pair using a loopback listener.
    async fn loopback_pair() -> (Connection, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let conn = Connection::new(server_stream, peer_addr).unwrap();
        (conn, client_stream)
    }

    #[tokio::test]
    async fn test_connection_initial_state() {
        let (conn, _client) = loopback_pair().await;
        assert_eq!(conn.state, ConnectionState::Handshaking);
        assert_eq!(conn.protocol_version, 0);
        assert!(!conn.is_encrypted());
        assert!(!conn.is_compressed());
    }

    #[tokio::test]
    async fn test_raw_packet_roundtrip() {
        let (mut server, mut client) = loopback_pair().await;

        // Client sends a framed packet: VarInt(len) + VarInt(packet_id) + body
        let packet_id: i32 = 0x00;
        let body = b"hello";

        // Build the inner payload (packet_id + body)
        let mut inner = BytesMut::new();
        varint::write_varint_buf(packet_id, &mut inner);
        inner.extend_from_slice(body);

        // Write as a frame from the client side
        frame::write_frame(&mut client, &inner).await.unwrap();
        client.flush().await.unwrap();

        // Server reads the raw packet
        let pkt = server.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x00);
        assert_eq!(&pkt.data[..], body);
    }

    #[tokio::test]
    async fn test_send_raw_and_read_back() {
        let (mut server, client) = loopback_pair().await;

        // Server sends a packet
        server.send_raw(0x01, b"pong").await.unwrap();
        server.flush().await.unwrap();

        // Read it back from the client side using frame codec
        let mut client_read = tokio::io::BufReader::new(client);
        let frame = frame::read_frame(&mut client_read, MAX_PACKET_SIZE)
            .await
            .unwrap();
        let mut buf = frame;
        let id = varint::read_varint_buf(&mut buf).unwrap();
        assert_eq!(id, 0x01);
        assert_eq!(&buf[..], b"pong");
    }

    #[tokio::test]
    async fn test_connection_state_display() {
        assert_eq!(ConnectionState::Handshaking.to_string(), "Handshaking");
        assert_eq!(ConnectionState::Status.to_string(), "Status");
        assert_eq!(ConnectionState::Login.to_string(), "Login");
        assert_eq!(ConnectionState::Configuration.to_string(), "Configuration");
        assert_eq!(ConnectionState::Play.to_string(), "Play");
    }

    #[tokio::test]
    async fn test_encrypted_roundtrip() {
        // Two connections: server-side and client-side both encrypt
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        // Enable encryption on both sides with same shared secret
        let secret = [0x42u8; 16];
        server.enable_encryption(&secret);
        client_conn.enable_encryption(&secret);

        assert!(server.is_encrypted());
        assert!(client_conn.is_encrypted());

        // Client sends an encrypted packet
        client_conn
            .send_raw(0x05, b"encrypted payload")
            .await
            .unwrap();
        client_conn.flush().await.unwrap();

        // Server reads and decrypts
        let pkt = server.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x05);
        assert_eq!(&pkt.data[..], b"encrypted payload");
    }

    #[tokio::test]
    async fn test_compressed_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        // Enable compression (threshold=64) on both sides
        server.enable_compression(64);
        client_conn.enable_compression(64);

        // Send a large payload that will be compressed
        let payload = vec![0xAB; 256];
        client_conn.send_raw(0x07, &payload).await.unwrap();
        client_conn.flush().await.unwrap();

        let pkt = server.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x07);
        assert_eq!(&pkt.data[..], &payload[..]);
    }

    #[tokio::test]
    async fn test_compressed_below_threshold() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        // Enable compression (threshold=256) — small packets stay uncompressed
        server.enable_compression(256);
        client_conn.enable_compression(256);

        // Small payload stays uncompressed (data_length=0)
        client_conn.send_raw(0x01, b"tiny").await.unwrap();
        client_conn.flush().await.unwrap();

        let pkt = server.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x01);
        assert_eq!(&pkt.data[..], b"tiny");
    }

    #[tokio::test]
    async fn test_encrypted_and_compressed_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        // Enable both encryption and compression
        let secret = [0x13u8; 16];
        server.enable_encryption(&secret);
        client_conn.enable_encryption(&secret);
        server.enable_compression(64);
        client_conn.enable_compression(64);

        // Large payload: encrypted + compressed
        let payload = vec![0xCD; 512];
        client_conn.send_raw(0x0A, &payload).await.unwrap();
        client_conn.flush().await.unwrap();

        let pkt = server.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x0A);
        assert_eq!(&pkt.data[..], &payload[..]);

        // Small payload: encrypted + uncompressed (below threshold)
        client_conn.send_raw(0x0B, b"small").await.unwrap();
        client_conn.flush().await.unwrap();

        let pkt2 = server.read_raw_packet().await.unwrap();
        assert_eq!(pkt2.id, 0x0B);
        assert_eq!(&pkt2.data[..], b"small");
    }

    // -----------------------------------------------------------------------
    // Generic typed packet I/O tests
    // -----------------------------------------------------------------------

    /// Minimal test packet implementing the `Packet` trait.
    #[derive(Debug, Clone, PartialEq)]
    struct TestPacket {
        value: i32,
    }

    impl Packet for TestPacket {
        const PACKET_ID: i32 = 0x42;

        fn decode(mut data: Bytes) -> Result<Self, PacketDecodeError> {
            use oxidized_codec::types::read_i32;
            let value = read_i32(&mut data)?;
            Ok(Self { value })
        }

        fn encode(&self) -> BytesMut {
            use oxidized_codec::types::write_i32;
            let mut buf = BytesMut::with_capacity(4);
            write_i32(&mut buf, self.value);
            buf
        }
    }

    #[tokio::test]
    async fn test_send_packet_roundtrip() {
        // Two Connection endpoints so both sides use encrypted/compressed pipeline
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        // Server sends a typed packet
        let pkt = TestPacket { value: 1_234_567 };
        server.send_packet(&pkt).await.unwrap();

        // Client reads it back as a raw packet, then decode via Connection helper
        let raw = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(raw.id, TestPacket::PACKET_ID);

        let decoded: TestPacket = Connection::decode_packet(&raw).unwrap();
        assert_eq!(decoded, pkt);
    }

    #[tokio::test]
    async fn test_send_packet_encrypted_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        let secret = [0x77u8; 16];
        server.enable_encryption(&secret);
        client_conn.enable_encryption(&secret);

        let pkt = TestPacket { value: -99 };
        server.send_packet(&pkt).await.unwrap();

        let raw = client_conn.read_raw_packet().await.unwrap();
        let decoded: TestPacket = Connection::decode_packet(&raw).unwrap();
        assert_eq!(decoded, pkt);
    }

    #[tokio::test]
    async fn test_send_packet_compressed_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        server.enable_compression(4);
        client_conn.enable_compression(4);

        let pkt = TestPacket { value: 42 };
        server.send_packet(&pkt).await.unwrap();

        let raw = client_conn.read_raw_packet().await.unwrap();
        let decoded: TestPacket = Connection::decode_packet(&raw).unwrap();
        assert_eq!(decoded, pkt);
    }

    #[tokio::test]
    async fn test_decode_packet_error_propagation() {
        // Decode with too-short data should produce a PacketDecodeError
        let raw = RawPacket {
            id: TestPacket::PACKET_ID,
            data: Bytes::from_static(&[0x00, 0x01]), // only 2 bytes, need 4
        };
        let result: Result<TestPacket, _> = Connection::decode_packet(&raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_connection_error_protocol_variant() {
        let pde = PacketDecodeError::InvalidData("test error".into());
        let ce: ConnectionError = pde.into();
        assert!(matches!(ce, ConnectionError::Protocol(_)));
        assert!(ce.to_string().contains("test error"));
    }

    // -----------------------------------------------------------------------
    // Connection::into_split tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_into_split_produces_reader_writer() {
        let (conn, _client) = loopback_pair().await;
        let addr = conn.remote_addr();
        let (reader, writer) = conn.into_split();
        assert_eq!(reader.remote_addr(), addr);
        assert_eq!(writer.remote_addr(), addr);
    }

    #[tokio::test]
    async fn test_split_reader_reads_packets() {
        let (conn, mut client) = loopback_pair().await;
        let (mut reader, _writer) = conn.into_split();

        // Client sends a framed packet
        let mut inner = BytesMut::new();
        varint::write_varint_buf(0x0A, &mut inner);
        inner.extend_from_slice(b"split read");
        frame::write_frame(&mut client, &inner).await.unwrap();
        client.flush().await.unwrap();

        let pkt = reader.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x0A);
        assert_eq!(&pkt.data[..], b"split read");
    }

    #[tokio::test]
    async fn test_split_writer_sends_packets() {
        let (conn, client) = loopback_pair().await;
        let (_reader, mut writer) = conn.into_split();

        writer.send_raw(0x0B, b"split write").await.unwrap();
        writer.flush().await.unwrap();

        let mut client_read = tokio::io::BufReader::new(client);
        let frame = frame::read_frame(&mut client_read, MAX_PACKET_SIZE)
            .await
            .unwrap();
        let mut buf = frame;
        let id = varint::read_varint_buf(&mut buf).unwrap();
        assert_eq!(id, 0x0B);
        assert_eq!(&buf[..], b"split write");
    }

    #[tokio::test]
    async fn test_split_encrypted_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        // Enable encryption on both sides
        let secret = [0x42u8; 16];
        server.enable_encryption(&secret);
        client_conn.enable_encryption(&secret);

        // Split the server connection
        let (mut reader, mut writer) = server.into_split();

        // Client sends encrypted → reader decrypts
        client_conn
            .send_raw(0x05, b"encrypted to reader")
            .await
            .unwrap();
        client_conn.flush().await.unwrap();

        let pkt = reader.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x05);
        assert_eq!(&pkt.data[..], b"encrypted to reader");

        // Writer encrypts → client decrypts
        writer
            .send_raw(0x06, b"encrypted from writer")
            .await
            .unwrap();
        writer.flush().await.unwrap();

        let pkt2 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt2.id, 0x06);
        assert_eq!(&pkt2.data[..], b"encrypted from writer");
    }

    #[tokio::test]
    async fn test_split_compressed_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        server.enable_compression(64);
        client_conn.enable_compression(64);

        let (mut reader, mut writer) = server.into_split();

        // Large payload (above threshold) — compressed
        let payload = vec![0xAB; 256];
        client_conn.send_raw(0x07, &payload).await.unwrap();
        client_conn.flush().await.unwrap();

        let pkt = reader.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x07);
        assert_eq!(&pkt.data[..], &payload[..]);

        // Writer sends large payload → client reads
        writer.send_raw(0x08, &payload).await.unwrap();
        writer.flush().await.unwrap();

        let pkt2 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt2.id, 0x08);
        assert_eq!(&pkt2.data[..], &payload[..]);
    }

    #[tokio::test]
    async fn test_split_encrypted_and_compressed_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        let secret = [0x13u8; 16];
        server.enable_encryption(&secret);
        client_conn.enable_encryption(&secret);
        server.enable_compression(64);
        client_conn.enable_compression(64);

        let (mut reader, mut writer) = server.into_split();

        // Large encrypted + compressed payload
        let payload = vec![0xCD; 512];
        client_conn.send_raw(0x09, &payload).await.unwrap();
        client_conn.flush().await.unwrap();

        let pkt = reader.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x09);
        assert_eq!(&pkt.data[..], &payload[..]);

        // Writer → client
        writer.send_raw(0x0A, &payload).await.unwrap();
        writer.flush().await.unwrap();

        let pkt2 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt2.id, 0x0A);
        assert_eq!(&pkt2.data[..], &payload[..]);

        // Small payload (below compression threshold, still encrypted)
        client_conn.send_raw(0x0B, b"small").await.unwrap();
        client_conn.flush().await.unwrap();

        let pkt3 = reader.read_raw_packet().await.unwrap();
        assert_eq!(pkt3.id, 0x0B);
        assert_eq!(&pkt3.data[..], b"small");
    }

    #[tokio::test]
    async fn test_split_no_encryption_no_compression() {
        let (conn, mut client) = loopback_pair().await;
        let (mut reader, mut writer) = conn.into_split();

        // Plaintext read
        let mut inner = BytesMut::new();
        varint::write_varint_buf(0x01, &mut inner);
        inner.extend_from_slice(b"plain");
        frame::write_frame(&mut client, &inner).await.unwrap();
        client.flush().await.unwrap();

        let pkt = reader.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x01);
        assert_eq!(&pkt.data[..], b"plain");

        // Plaintext write
        writer.send_raw(0x02, b"response").await.unwrap();
        writer.flush().await.unwrap();

        let mut client_read = tokio::io::BufReader::new(client);
        let frame = frame::read_frame(&mut client_read, MAX_PACKET_SIZE)
            .await
            .unwrap();
        let mut buf = frame;
        let id = varint::read_varint_buf(&mut buf).unwrap();
        assert_eq!(id, 0x02);
        assert_eq!(&buf[..], b"response");
    }

    #[tokio::test]
    async fn test_split_writer_send_packet() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        let (_reader, mut writer) = server.into_split();

        let pkt = TestPacket { value: 9999 };
        writer.send_packet(&pkt).await.unwrap();

        let raw = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(raw.id, TestPacket::PACKET_ID);
        let decoded: TestPacket = Connection::decode_packet(&raw).unwrap();
        assert_eq!(decoded, pkt);
    }

    #[tokio::test]
    async fn test_split_batch_buf_initial_capacity() {
        let (conn, _client) = loopback_pair().await;
        let (_reader, writer) = conn.into_split();
        // batch_buf should be pre-allocated
        assert!(writer.batch_buf_len() == 0);
    }

    #[tokio::test]
    async fn test_split_debug_formatting() {
        let (conn, _client) = loopback_pair().await;
        let (reader, writer) = conn.into_split();
        // Ensure Debug doesn't panic
        let _ = format!("{reader:?}");
        let _ = format!("{writer:?}");
    }

    // -----------------------------------------------------------------------
    // Batch encoding (R4.3) tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_batch_single_packet() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        let (_reader, mut writer) = server.into_split();

        // Encode one packet into batch
        writer.encode_to_batch(0x0B, b"hello batch").unwrap();
        assert!(writer.batch_buf_len() > 0);

        // Flush the batch
        writer.flush_batch().await.unwrap();
        assert_eq!(writer.batch_buf_len(), 0);

        // Client reads it back
        let pkt = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt.id, 0x0B);
        assert_eq!(&pkt.data[..], b"hello batch");
    }

    #[tokio::test]
    async fn test_batch_multiple_packets() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        let (_reader, mut writer) = server.into_split();

        // Encode three packets into one batch
        writer.encode_to_batch(0x01, b"first").unwrap();
        writer.encode_to_batch(0x02, b"second").unwrap();
        writer.encode_to_batch(0x03, b"third").unwrap();

        // Single flush for all three
        writer.flush_batch().await.unwrap();
        assert_eq!(writer.batch_buf_len(), 0);

        // Client reads all three
        let pkt1 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt1.id, 0x01);
        assert_eq!(&pkt1.data[..], b"first");

        let pkt2 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt2.id, 0x02);
        assert_eq!(&pkt2.data[..], b"second");

        let pkt3 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt3.id, 0x03);
        assert_eq!(&pkt3.data[..], b"third");
    }

    #[tokio::test]
    async fn test_batch_empty_flush_is_noop() {
        let (conn, _client) = loopback_pair().await;
        let (_reader, mut writer) = conn.into_split();

        // Flushing an empty batch should not error
        writer.flush_batch().await.unwrap();
        assert_eq!(writer.batch_buf_len(), 0);
    }

    #[tokio::test]
    async fn test_batch_clears_after_flush() {
        let (conn, _client) = loopback_pair().await;
        let (_reader, mut writer) = conn.into_split();

        writer.encode_to_batch(0x01, b"data").unwrap();
        assert!(writer.batch_buf_len() > 0);

        writer.flush_batch().await.unwrap();
        assert_eq!(writer.batch_buf_len(), 0);

        // Second batch works too
        writer.encode_to_batch(0x02, b"more data").unwrap();
        assert!(writer.batch_buf_len() > 0);
    }

    #[tokio::test]
    async fn test_batch_encrypted_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        let secret = [0x55u8; 16];
        server.enable_encryption(&secret);
        client_conn.enable_encryption(&secret);

        let (_reader, mut writer) = server.into_split();

        // Batch two encrypted packets
        writer.encode_to_batch(0x10, b"encrypted one").unwrap();
        writer.encode_to_batch(0x11, b"encrypted two").unwrap();
        writer.flush_batch().await.unwrap();

        let pkt1 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt1.id, 0x10);
        assert_eq!(&pkt1.data[..], b"encrypted one");

        let pkt2 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt2.id, 0x11);
        assert_eq!(&pkt2.data[..], b"encrypted two");
    }

    #[tokio::test]
    async fn test_batch_compressed_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        server.enable_compression(64);
        client_conn.enable_compression(64);

        let (_reader, mut writer) = server.into_split();

        // Large payload (above threshold) — compressed
        let payload = vec![0xAB; 256];
        writer.encode_to_batch(0x20, &payload).unwrap();
        // Small payload (below threshold) — uncompressed but with data_length=0
        writer.encode_to_batch(0x21, b"tiny").unwrap();
        writer.flush_batch().await.unwrap();

        let pkt1 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt1.id, 0x20);
        assert_eq!(&pkt1.data[..], &payload[..]);

        let pkt2 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt2.id, 0x21);
        assert_eq!(&pkt2.data[..], b"tiny");
    }

    #[tokio::test]
    async fn test_batch_encrypted_and_compressed_roundtrip() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let mut server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        let secret = [0xAAu8; 16];
        server.enable_encryption(&secret);
        client_conn.enable_encryption(&secret);
        server.enable_compression(64);
        client_conn.enable_compression(64);

        let (_reader, mut writer) = server.into_split();

        // Large (encrypted + compressed) + small (encrypted, below threshold)
        let large = vec![0xCD; 512];
        writer.encode_to_batch(0x30, &large).unwrap();
        writer.encode_to_batch(0x31, b"small").unwrap();
        writer.flush_batch().await.unwrap();

        let pkt1 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt1.id, 0x30);
        assert_eq!(&pkt1.data[..], &large[..]);

        let pkt2 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt2.id, 0x31);
        assert_eq!(&pkt2.data[..], b"small");
    }

    #[tokio::test]
    async fn test_batch_multiple_flush_cycles() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_handle = tokio::spawn(async move { TcpStream::connect(addr).await.unwrap() });
        let (server_stream, peer_addr) = listener.accept().await.unwrap();
        let client_stream = client_handle.await.unwrap();

        let server = Connection::new(server_stream, peer_addr).unwrap();
        let mut client_conn =
            Connection::new(client_stream, "127.0.0.1:0".parse().unwrap()).unwrap();

        let (_reader, mut writer) = server.into_split();

        // Cycle 1
        writer.encode_to_batch(0x01, b"cycle1").unwrap();
        writer.flush_batch().await.unwrap();

        // Cycle 2
        writer.encode_to_batch(0x02, b"cycle2a").unwrap();
        writer.encode_to_batch(0x03, b"cycle2b").unwrap();
        writer.flush_batch().await.unwrap();

        let pkt1 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt1.id, 0x01);
        assert_eq!(&pkt1.data[..], b"cycle1");

        let pkt2 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt2.id, 0x02);
        assert_eq!(&pkt2.data[..], b"cycle2a");

        let pkt3 = client_conn.read_raw_packet().await.unwrap();
        assert_eq!(pkt3.id, 0x03);
        assert_eq!(&pkt3.data[..], b"cycle2b");
    }

    // -------------------------------------------------------------------
    // ConnectionStateMachine tests (no network I/O required)
    // -------------------------------------------------------------------

    #[test]
    fn test_state_machine_initial_state() {
        let sm = ConnectionStateMachine::new();
        assert_eq!(sm.state(), ConnectionState::Handshaking);
        assert!(!sm.is_encrypted());
        assert!(!sm.is_compressed());
    }

    #[test]
    fn test_state_machine_default() {
        let sm = ConnectionStateMachine::default();
        assert_eq!(sm.state(), ConnectionState::Handshaking);
    }

    #[test]
    fn test_state_machine_handshaking_to_status() {
        let mut sm = ConnectionStateMachine::new();
        sm.transition(ConnectionState::Status).unwrap();
        assert_eq!(sm.state(), ConnectionState::Status);
    }

    #[test]
    fn test_state_machine_handshaking_to_login() {
        let mut sm = ConnectionStateMachine::new();
        sm.transition(ConnectionState::Login).unwrap();
        assert_eq!(sm.state(), ConnectionState::Login);
    }

    #[test]
    fn test_state_machine_full_login_flow() {
        let mut sm = ConnectionStateMachine::new();

        sm.transition(ConnectionState::Login).unwrap();
        assert_eq!(sm.state(), ConnectionState::Login);

        sm.transition(ConnectionState::Configuration).unwrap();
        assert_eq!(sm.state(), ConnectionState::Configuration);

        sm.transition(ConnectionState::Play).unwrap();
        assert_eq!(sm.state(), ConnectionState::Play);
    }

    #[test]
    fn test_state_machine_invalid_handshaking_to_play() {
        let mut sm = ConnectionStateMachine::new();
        let err = sm.transition(ConnectionState::Play).unwrap_err();
        assert_eq!(err.from, ConnectionState::Handshaking);
        assert_eq!(err.to, ConnectionState::Play);
        // State should remain unchanged after invalid transition.
        assert_eq!(sm.state(), ConnectionState::Handshaking);
    }

    #[test]
    fn test_state_machine_invalid_handshaking_to_configuration() {
        let mut sm = ConnectionStateMachine::new();
        let err = sm.transition(ConnectionState::Configuration).unwrap_err();
        assert_eq!(err.from, ConnectionState::Handshaking);
        assert_eq!(err.to, ConnectionState::Configuration);
    }

    #[test]
    fn test_state_machine_invalid_status_to_login() {
        let mut sm = ConnectionStateMachine::new();
        sm.transition(ConnectionState::Status).unwrap();
        let err = sm.transition(ConnectionState::Login).unwrap_err();
        assert_eq!(err.from, ConnectionState::Status);
        assert_eq!(err.to, ConnectionState::Login);
    }

    #[test]
    fn test_state_machine_invalid_login_to_play() {
        let mut sm = ConnectionStateMachine::new();
        sm.transition(ConnectionState::Login).unwrap();
        let err = sm.transition(ConnectionState::Play).unwrap_err();
        assert_eq!(err.from, ConnectionState::Login);
        assert_eq!(err.to, ConnectionState::Play);
    }

    #[test]
    fn test_state_machine_invalid_play_to_anything() {
        let mut sm = ConnectionStateMachine::new();
        sm.transition(ConnectionState::Login).unwrap();
        sm.transition(ConnectionState::Configuration).unwrap();
        sm.transition(ConnectionState::Play).unwrap();

        // Play is a terminal state — no valid transitions out.
        for target in [
            ConnectionState::Handshaking,
            ConnectionState::Status,
            ConnectionState::Login,
            ConnectionState::Configuration,
            ConnectionState::Play,
        ] {
            assert!(sm.transition(target).is_err());
        }
    }

    #[test]
    fn test_state_machine_encryption_compression_flags() {
        let mut sm = ConnectionStateMachine::new();
        assert!(!sm.is_encrypted());
        assert!(!sm.is_compressed());

        sm.set_encrypted();
        assert!(sm.is_encrypted());
        assert!(!sm.is_compressed());

        sm.set_compressed();
        assert!(sm.is_encrypted());
        assert!(sm.is_compressed());
    }

    #[test]
    fn test_state_machine_is_valid_transition_exhaustive() {
        use ConnectionState::*;
        let all = [Handshaking, Status, Login, Configuration, Play];

        let valid = [
            (Handshaking, Status),
            (Handshaking, Login),
            (Login, Configuration),
            (Configuration, Play),
        ];

        for &from in &all {
            for &to in &all {
                let expected = valid.contains(&(from, to));
                assert_eq!(
                    ConnectionStateMachine::is_valid_transition(from, to),
                    expected,
                    "is_valid_transition({from}, {to}) should be {expected}"
                );
            }
        }
    }

    #[test]
    fn test_state_machine_invalid_transition_display() {
        let err = InvalidTransition {
            from: ConnectionState::Handshaking,
            to: ConnectionState::Play,
        };
        assert_eq!(
            err.to_string(),
            "invalid state transition: Handshaking -> Play"
        );
    }
}
