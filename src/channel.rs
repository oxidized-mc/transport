//! Channel types and constants for the reader/writer task pair model.
//!
//! Defines the packet wrappers that flow through bounded `mpsc` channels
//! between the reader task, writer task, and game logic. Also defines
//! the capacity and rate-limiting constants specified by
//! ADR-006 (Network I/O).

use bytes::Bytes;

// ---------------------------------------------------------------------------
// Inbound packets (reader → game logic)
// ---------------------------------------------------------------------------

/// A decoded inbound packet received from a client.
///
/// Produced by the reader task after decryption, decompression, and
/// frame decoding. Sent to the game logic through a bounded `mpsc`
/// channel.
#[derive(Debug, Clone)]
pub struct InboundPacket {
    /// Packet ID (VarInt on the wire, decoded to `i32`).
    pub id: i32,
    /// Packet payload (decompressed, decrypted).
    pub data: Bytes,
}

// ---------------------------------------------------------------------------
// Outbound packets (game logic → writer)
// ---------------------------------------------------------------------------

/// An outbound packet to be sent to a client.
///
/// Produced by game logic (handlers, broadcast, keepalive) and queued
/// on the outbound `mpsc` channel. The writer task encodes, compresses,
/// encrypts, and flushes it.
#[derive(Debug, Clone)]
pub struct OutboundPacket {
    /// Packet ID.
    pub id: i32,
    /// Pre-encoded packet payload (before compression/encryption).
    pub data: Bytes,
}

// ---------------------------------------------------------------------------
// Constants (ADR-006)
// ---------------------------------------------------------------------------

/// Channel capacity for inbound packets (reader → game logic).
///
/// Backpressure: when the game logic is slow to consume packets, the
/// reader task blocks on `send().await`, which stops reading from TCP,
/// triggering TCP flow control on the client.
pub const INBOUND_CHANNEL_CAPACITY: usize = 128;

/// Channel capacity for outbound packets (game logic → writer).
///
/// Sized for burst traffic (join sequence, chunk loading). If the
/// writer cannot drain fast enough (slow client), senders see
/// backpressure or `try_send` failures.
pub const OUTBOUND_CHANNEL_CAPACITY: usize = 512;

/// Maximum packets a client may send per tick window (50 ms).
///
/// Exceeding this limit triggers a forced disconnect. Vanilla clients
/// send ~5–10 packets per tick under normal gameplay.
pub const MAX_PACKETS_PER_TICK: u32 = 500;

/// Maximum combined buffer memory per connection (256 KB).
///
/// Includes the writer's batch buffer and any in-flight data. If
/// exceeded, the connection is terminated to prevent memory exhaustion.
pub const MAX_CONNECTION_MEMORY: usize = 256 * 1024;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[test]
    fn test_inbound_packet_fields() {
        let pkt = InboundPacket {
            id: 0x0E,
            data: Bytes::from_static(b"hello"),
        };
        assert_eq!(pkt.id, 0x0E);
        assert_eq!(&pkt.data[..], b"hello");
    }

    #[test]
    fn test_outbound_packet_fields() {
        let pkt = OutboundPacket {
            id: 0x24,
            data: Bytes::from_static(b"world"),
        };
        assert_eq!(pkt.id, 0x24);
        assert_eq!(&pkt.data[..], b"world");
    }

    #[tokio::test]
    async fn test_inbound_channel_send_recv() {
        let (tx, mut rx) = mpsc::channel::<InboundPacket>(INBOUND_CHANNEL_CAPACITY);
        let pkt = InboundPacket {
            id: 0x01,
            data: Bytes::from_static(b"test"),
        };
        tx.send(pkt.clone()).await.unwrap();
        let received = rx.recv().await.unwrap();
        assert_eq!(received.id, 0x01);
        assert_eq!(received.data, pkt.data);
    }

    #[tokio::test]
    async fn test_outbound_channel_send_recv() {
        let (tx, mut rx) = mpsc::channel::<OutboundPacket>(OUTBOUND_CHANNEL_CAPACITY);
        let pkt = OutboundPacket {
            id: 0x42,
            data: Bytes::from_static(b"payload"),
        };
        tx.send(pkt.clone()).await.unwrap();
        let received = rx.recv().await.unwrap();
        assert_eq!(received.id, 0x42);
        assert_eq!(received.data, pkt.data);
    }

    #[tokio::test]
    async fn test_outbound_try_send_full_channel() {
        // Channel capacity 1 to easily fill it
        let (tx, _rx) = mpsc::channel::<OutboundPacket>(1);
        let pkt = OutboundPacket {
            id: 0x01,
            data: Bytes::from_static(b"a"),
        };
        tx.send(pkt).await.unwrap();

        // Second try_send should fail (channel full)
        let pkt2 = OutboundPacket {
            id: 0x02,
            data: Bytes::from_static(b"b"),
        };
        assert!(tx.try_send(pkt2).is_err());
    }

    #[test]
    fn test_constants_match_adr006() {
        assert_eq!(INBOUND_CHANNEL_CAPACITY, 128);
        assert_eq!(OUTBOUND_CHANNEL_CAPACITY, 512);
        assert_eq!(MAX_PACKETS_PER_TICK, 500);
        assert_eq!(MAX_CONNECTION_MEMORY, 256 * 1024);
    }
}
