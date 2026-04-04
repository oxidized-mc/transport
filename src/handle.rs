//! Connection handle — the send-side API for the task pair model.
//!
//! A [`ConnectionHandle`] provides a typed interface for queueing packets
//! on the outbound channel without direct access to the TCP stream. The
//! writer task drains the channel and flushes packets to the network.
//!
//! See ADR-006 (Network I/O) for the
//! reader/writer task pair architecture.

use std::net::SocketAddr;

use bytes::Bytes;
use thiserror::Error;
use tokio::sync::mpsc;

use crate::channel::OutboundPacket;
use oxidized_codec::packet::Packet;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from [`ConnectionHandle`] send operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HandleError {
    /// The outbound channel is full (slow client) or the writer task has
    /// exited (channel closed).
    #[error("outbound channel closed or full")]
    ChannelClosed,
}

// ---------------------------------------------------------------------------
// ConnectionHandle
// ---------------------------------------------------------------------------

/// Handle to an active connection's outbound channel.
///
/// Provides a typed interface for sending packets without needing direct
/// access to the TCP stream. Sending through the handle queues the
/// packet on the outbound channel; the writer task flushes it.
///
/// Cloning a handle creates another sender to the same outbound channel.
#[derive(Debug, Clone)]
pub struct ConnectionHandle {
    outbound_tx: mpsc::Sender<OutboundPacket>,
    addr: SocketAddr,
}

impl ConnectionHandle {
    /// Creates a new handle from an outbound channel sender and address.
    pub fn new(outbound_tx: mpsc::Sender<OutboundPacket>, addr: SocketAddr) -> Self {
        Self { outbound_tx, addr }
    }

    /// Queues a typed packet for sending to the client.
    ///
    /// The packet is encoded via [`Packet::encode`] and queued on the
    /// outbound channel. The writer task handles compression, encryption,
    /// and flushing.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError::ChannelClosed`] if the outbound channel is
    /// closed (writer task exited).
    pub async fn send_packet<P: Packet>(&self, pkt: &P) -> Result<(), HandleError> {
        let body = pkt.encode();
        self.send_raw(P::PACKET_ID, body.freeze()).await
    }

    /// Non-blocking send of a typed packet.
    ///
    /// Returns immediately if the channel is full or closed.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError::ChannelClosed`] if the channel is full or
    /// the writer task has exited.
    pub fn try_send_packet<P: Packet>(&self, pkt: &P) -> Result<(), HandleError> {
        let body = pkt.encode();
        self.try_send_raw(P::PACKET_ID, body.freeze())
    }

    /// Queues a pre-encoded raw packet on the outbound channel.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError::ChannelClosed`] if the channel is closed.
    pub async fn send_raw(&self, id: i32, data: Bytes) -> Result<(), HandleError> {
        self.outbound_tx
            .send(OutboundPacket { id, data })
            .await
            .map_err(|_| HandleError::ChannelClosed)
    }

    /// Non-blocking raw packet send.
    ///
    /// Returns immediately if the channel is full or closed.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError::ChannelClosed`] if the channel is full or
    /// the writer task has exited.
    pub fn try_send_raw(&self, id: i32, data: Bytes) -> Result<(), HandleError> {
        self.outbound_tx
            .try_send(OutboundPacket { id, data })
            .map_err(|_| HandleError::ChannelClosed)
    }

    /// Returns the remote peer address.
    pub fn remote_addr(&self) -> SocketAddr {
        self.addr
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use bytes::{Bytes, BytesMut};
    use oxidized_codec::packet::{Packet, PacketDecodeError};
    use oxidized_codec::types::{read_i32, write_i32};

    /// Minimal test packet for handle tests.
    #[derive(Debug, Clone, PartialEq)]
    struct TestPacket {
        value: i32,
    }

    impl Packet for TestPacket {
        const PACKET_ID: i32 = 0x99;

        fn decode(mut data: Bytes) -> Result<Self, PacketDecodeError> {
            let value = read_i32(&mut data)?;
            Ok(Self { value })
        }

        fn encode(&self) -> BytesMut {
            let mut buf = BytesMut::with_capacity(4);
            write_i32(&mut buf, self.value);
            buf
        }
    }

    fn test_addr() -> SocketAddr {
        "127.0.0.1:25565".parse().unwrap()
    }

    #[tokio::test]
    async fn test_send_packet_queues_on_channel() {
        let (tx, mut rx) = mpsc::channel(16);
        let handle = ConnectionHandle::new(tx, test_addr());

        let pkt = TestPacket { value: 42 };
        handle.send_packet(&pkt).await.unwrap();

        let outbound = rx.recv().await.unwrap();
        assert_eq!(outbound.id, TestPacket::PACKET_ID);

        // Decode the body to verify it matches
        let decoded: TestPacket = Packet::decode(outbound.data).unwrap();
        assert_eq!(decoded, pkt);
    }

    #[tokio::test]
    async fn test_try_send_packet_success() {
        let (tx, mut rx) = mpsc::channel(16);
        let handle = ConnectionHandle::new(tx, test_addr());

        let pkt = TestPacket { value: -1 };
        handle.try_send_packet(&pkt).unwrap();

        let outbound = rx.recv().await.unwrap();
        assert_eq!(outbound.id, TestPacket::PACKET_ID);
    }

    #[tokio::test]
    async fn test_try_send_packet_channel_full() {
        let (tx, _rx) = mpsc::channel(1);
        let handle = ConnectionHandle::new(tx, test_addr());

        // Fill the channel
        handle
            .send_raw(0x01, Bytes::from_static(b"a"))
            .await
            .unwrap();

        // try_send should fail now
        let pkt = TestPacket { value: 0 };
        let result = handle.try_send_packet(&pkt);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_raw_queues_on_channel() {
        let (tx, mut rx) = mpsc::channel(16);
        let handle = ConnectionHandle::new(tx, test_addr());

        handle
            .send_raw(0x42, Bytes::from_static(b"raw data"))
            .await
            .unwrap();

        let outbound = rx.recv().await.unwrap();
        assert_eq!(outbound.id, 0x42);
        assert_eq!(&outbound.data[..], b"raw data");
    }

    #[tokio::test]
    async fn test_send_raw_channel_closed() {
        let (tx, rx) = mpsc::channel(16);
        let handle = ConnectionHandle::new(tx, test_addr());

        // Drop receiver to close the channel
        drop(rx);

        let result = handle.send_raw(0x01, Bytes::from_static(b"data")).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_try_send_raw_channel_closed() {
        let (tx, rx) = mpsc::channel::<OutboundPacket>(16);
        let handle = ConnectionHandle::new(tx, "127.0.0.1:1".parse().unwrap());

        drop(rx);

        let result = handle.try_send_raw(0x01, Bytes::from_static(b"data"));
        assert!(result.is_err());
    }

    #[test]
    fn test_remote_addr() {
        let (tx, _rx) = mpsc::channel(1);
        let addr: SocketAddr = "192.168.1.1:12345".parse().unwrap();
        let handle = ConnectionHandle::new(tx, addr);
        assert_eq!(handle.remote_addr(), addr);
    }

    #[tokio::test]
    async fn test_handle_clone_shares_channel() {
        let (tx, mut rx) = mpsc::channel(16);
        let handle = ConnectionHandle::new(tx, test_addr());
        let handle2 = handle.clone();

        handle
            .send_raw(0x01, Bytes::from_static(b"from_1"))
            .await
            .unwrap();
        handle2
            .send_raw(0x02, Bytes::from_static(b"from_2"))
            .await
            .unwrap();

        let pkt1 = rx.recv().await.unwrap();
        let pkt2 = rx.recv().await.unwrap();
        assert_eq!(pkt1.id, 0x01);
        assert_eq!(pkt2.id, 0x02);
    }

    /// Verifies that `try_send_raw` returns an error when the outbound
    /// channel is full, simulating broadcast backpressure (R4.9).
    /// In the play loop, broadcasts use `try_send_raw()` — when the
    /// channel is full (slow client), the connection is disconnected.
    #[tokio::test]
    async fn test_broadcast_channel_full_disconnects() {
        // Channel capacity = 2 to easily fill it
        let (tx, _rx) = mpsc::channel(2);
        let handle = ConnectionHandle::new(tx, test_addr());

        // Fill the channel
        handle
            .send_raw(0x10, Bytes::from_static(b"broadcast_1"))
            .await
            .unwrap();
        handle
            .send_raw(0x10, Bytes::from_static(b"broadcast_2"))
            .await
            .unwrap();

        // Simulate broadcast: try_send_raw should fail (channel full)
        let result = handle.try_send_raw(0x10, Bytes::from_static(b"broadcast_3"));
        assert!(
            result.is_err(),
            "try_send_raw should fail when channel is full (slow client)"
        );
    }

    /// Verifies that `try_send_raw` succeeds when the channel has space,
    /// ensuring normal clients are not affected by the channel-full check.
    #[tokio::test]
    async fn test_broadcast_succeeds_with_available_capacity() {
        let (tx, mut rx) = mpsc::channel(16);
        let handle = ConnectionHandle::new(tx, test_addr());

        // Send several broadcast packets — should all succeed
        for i in 0..10 {
            handle
                .try_send_raw(0x10, Bytes::from(vec![i as u8]))
                .unwrap();
        }

        // Verify they're all queued
        for i in 0..10 {
            let pkt = rx.recv().await.unwrap();
            assert_eq!(pkt.id, 0x10);
            assert_eq!(&pkt.data[..], &[i as u8]);
        }
    }
}
