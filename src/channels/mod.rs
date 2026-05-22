//! Channel pub/sub bus.
//!
//! A tiny in-process publish/subscribe bus for coordinating between the main
//! agent and any sub-agents it spawns. Ported from the AgentCpp `ChannelBus`.
//!
//! - Channels are auto-created on first use.
//! - Each channel keeps a ring buffer of the most recent N messages
//!   (default 256).
//! - Messages stay in memory for the lifetime of the process — there is no
//!   on-disk persistence, this is purely for run-scoped coordination.
//! - The bus is a process-wide singleton and is thread-safe.

use chrono::Utc;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

/// One message published on a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMessage {
    /// Monotonically increasing id, scoped to the channel.
    pub id: u64,
    /// Wall-clock epoch milliseconds when the message was published.
    pub epoch_ms: i64,
    /// Free-form sender label (e.g. "main", "task-d1").
    pub sender: String,
    /// Body of the message.
    pub text: String,
}

/// Per-channel statistics for `list()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelInfo {
    pub name: String,
    pub message_count: usize,
    pub latest_id: u64,
}

#[derive(Default)]
struct Channel {
    messages: Vec<ChannelMessage>,
    next_id: u64,
}

/// Process-wide singleton.
pub struct ChannelBus {
    inner: Mutex<BusInner>,
}

struct BusInner {
    channels: HashMap<String, Channel>,
    ring_size: usize,
}

impl ChannelBus {
    fn new() -> Self {
        Self {
            inner: Mutex::new(BusInner {
                channels: HashMap::new(),
                ring_size: 256,
            }),
        }
    }

    /// Global singleton handle. Use this from anywhere.
    pub fn instance() -> &'static ChannelBus {
        static BUS: Lazy<ChannelBus> = Lazy::new(ChannelBus::new);
        &BUS
    }

    /// Configure the per-channel ring buffer size. Minimum 1.
    pub fn set_ring_size(&self, n: usize) {
        let mut inner = self.inner.lock().expect("ChannelBus mutex poisoned");
        inner.ring_size = n.max(1);
    }

    /// Publish a message; returns the assigned id.
    pub fn publish(&self, channel: &str, sender: &str, text: &str) -> u64 {
        let mut inner = self.inner.lock().expect("ChannelBus mutex poisoned");
        let ring_size = inner.ring_size;
        let c = inner.channels.entry(channel.to_string()).or_default();
        // next_id starts at 0; bump first so first id is 1 (matches AgentCpp behaviour).
        c.next_id += 1;
        let msg = ChannelMessage {
            id: c.next_id,
            epoch_ms: Utc::now().timestamp_millis(),
            sender: sender.to_string(),
            text: text.to_string(),
        };
        let assigned = msg.id;
        c.messages.push(msg);
        if c.messages.len() > ring_size {
            let over = c.messages.len() - ring_size;
            c.messages.drain(..over);
        }
        assigned
    }

    /// Return every message on `channel` with id > since_id, in order.
    pub fn read(&self, channel: &str, since_id: u64) -> Vec<ChannelMessage> {
        let inner = self.inner.lock().expect("ChannelBus mutex poisoned");
        match inner.channels.get(channel) {
            None => Vec::new(),
            Some(c) => c
                .messages
                .iter()
                .filter(|m| m.id > since_id)
                .cloned()
                .collect(),
        }
    }

    /// List every known channel (sorted by name) with its size and latest id.
    pub fn list(&self) -> Vec<ChannelInfo> {
        let inner = self.inner.lock().expect("ChannelBus mutex poisoned");
        let mut out: Vec<ChannelInfo> = inner
            .channels
            .iter()
            .map(|(name, c)| ChannelInfo {
                name: name.clone(),
                message_count: c.messages.len(),
                latest_id: c.messages.last().map(|m| m.id).unwrap_or(0),
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_then_read_returns_only_newer_messages() {
        let bus = ChannelBus::new();
        let id1 = bus.publish("c", "main", "hello");
        let id2 = bus.publish("c", "main", "world");
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        let all = bus.read("c", 0);
        assert_eq!(all.len(), 2);
        let only_new = bus.read("c", id1);
        assert_eq!(only_new.len(), 1);
        assert_eq!(only_new[0].text, "world");
    }

    #[test]
    fn ring_buffer_drops_oldest() {
        let bus = ChannelBus::new();
        bus.set_ring_size(2);
        bus.publish("c", "main", "a");
        bus.publish("c", "main", "b");
        bus.publish("c", "main", "c");
        let msgs = bus.read("c", 0);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text, "b");
        assert_eq!(msgs[1].text, "c");
    }

    #[test]
    fn list_returns_known_channels() {
        let bus = ChannelBus::new();
        bus.publish("alpha", "main", "x");
        bus.publish("beta", "main", "y");
        let infos = bus.list();
        let names: Vec<_> = infos.iter().map(|i| i.name.clone()).collect();
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }
}
