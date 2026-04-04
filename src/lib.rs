//! Minecraft protocol transport — Connection state machine, async reader/writer, framing pipeline.
//!
//! Groups the low-level protocol concerns that operate below the packet layer:
//! - [`connection`] — Per-client TCP stream, state machine, packet framing
//! - [`channel`] — Channel types and constants for the reader/writer task pair
//! - [`handle`] — Connection handle API for the outbound channel

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod channel;
pub mod connection;
pub mod handle;
