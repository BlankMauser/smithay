//! Selection offers follow keyboard focus: a client that lost the focus hears
//! that it has no selection and cannot read the offer it held, an offer is
//! stale once a newer selection was offered, and data-control devices (which
//! never need focus) are not affected.
//!
//! Real `wayland-client` connections over socket pairs, dispatched on the
//! test's thread.

use std::os::unix::io::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;

use rustix::pipe::{PipeFlags, pipe_with};
use wayland_client::protocol::{
    wl_data_device::{self, WlDataDevice},
    wl_data_device_manager::WlDataDeviceManager,
    wl_data_offer::WlDataOffer,
    wl_registry::{self, WlRegistry},
    wl_seat::WlSeat,
};
use wayland_client::{Connection, Dispatch, EventQueue, QueueHandle, delegate_noop, event_created_child};
use wayland_protocols::ext::data_control::v1::client::{
    ext_data_control_device_v1::{self, ExtDataControlDeviceV1},
    ext_data_control_manager_v1::ExtDataControlManagerV1,
    ext_data_control_offer_v1::ExtDataControlOfferV1,
};
use wayland_protocols::wp::primary_selection::zv1::client::{
    zwp_primary_selection_device_manager_v1::ZwpPrimarySelectionDeviceManagerV1,
    zwp_primary_selection_device_v1::{self, ZwpPrimarySelectionDeviceV1},
    zwp_primary_selection_offer_v1::ZwpPrimarySelectionOfferV1,
};
use wayland_protocols_wlr::data_control::v1::client::{
    zwlr_data_control_device_v1::{self, ZwlrDataControlDeviceV1},
    zwlr_data_control_manager_v1::ZwlrDataControlManagerV1,
    zwlr_data_control_offer_v1::ZwlrDataControlOfferV1,
    zwlr_data_control_source_v1::{self, ZwlrDataControlSourceV1},
};
use wayland_server::backend::{ClientData, ClientId, DisconnectReason};
use wayland_server::protocol::wl_surface::WlSurface;
use wayland_server::{Client, Display, DisplayHandle};

use crate::input::pointer::CursorImageStatus;
use crate::input::{Seat, SeatHandler, SeatState};
use crate::wayland::compositor::{CompositorClientState, CompositorHandler, CompositorState};
use crate::wayland::pointer_constraints::PointerConstraintsHandler;

use super::SelectionTarget;
use super::data_device::{
    DataDeviceHandler, DataDeviceState, WaylandDndGrabHandler, set_data_device_focus,
    set_data_device_selection,
};
use super::primary_selection::{
    PrimarySelectionHandler, PrimarySelectionState, set_primary_focus, set_primary_selection,
};
use super::{SelectionHandler, ext_data_control, wlr_data_control};

// The compositor.

struct State {
    seat_state: SeatState<State>,
    data_device: DataDeviceState,
    primary: PrimarySelectionState,
    wlr: wlr_data_control::DataControlState,
    ext: ext_data_control::DataControlState,
    /// Every read of the compositor's own selection that reached
    /// [`SelectionHandler::send_selection`].
    reads: Vec<(SelectionTarget, String)>,
}

impl SeatHandler for State {
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }
    fn focus_changed(&mut self, _: &Seat<Self>, _: Option<&WlSurface>) {}
    fn cursor_image(&mut self, _: &Seat<Self>, _: CursorImageStatus) {}
}

impl CompositorHandler for State {
    fn compositor_state(&mut self) -> &mut CompositorState {
        unreachable!("no surfaces in these tests")
    }
    fn client_compositor_state<'a>(&self, _: &'a Client) -> &'a CompositorClientState {
        unreachable!("no surfaces in these tests")
    }
    fn commit(&mut self, _: &WlSurface) {}
}

impl PointerConstraintsHandler for State {}

impl SelectionHandler for State {
    type SelectionUserData = ();

    /// Answer with the requested MIME type as the data.
    fn send_selection(&mut self, ty: SelectionTarget, mime_type: String, fd: OwnedFd, _: Seat<Self>, _: &()) {
        rustix::io::write(&fd, mime_type.as_bytes()).unwrap();
        self.reads.push((ty, mime_type));
    }
}

impl WaylandDndGrabHandler for State {}

impl DataDeviceHandler for State {
    fn data_device_state(&mut self) -> &mut DataDeviceState {
        &mut self.data_device
    }
}

impl PrimarySelectionHandler for State {
    fn primary_selection_state(&mut self) -> &mut PrimarySelectionState {
        &mut self.primary
    }
}

impl wlr_data_control::DataControlHandler for State {
    fn data_control_state(&mut self) -> &mut wlr_data_control::DataControlState {
        &mut self.wlr
    }
}

impl ext_data_control::DataControlHandler for State {
    fn data_control_state(&mut self) -> &mut ext_data_control::DataControlState {
        &mut self.ext
    }
}

crate::delegate_dispatch2!(State);

struct NoData;

impl ClientData for NoData {
    fn initialized(&self, _: ClientId) {}
    fn disconnected(&self, _: ClientId, _: DisconnectReason) {}
}

struct Compositor {
    display: Display<State>,
    dh: DisplayHandle,
    state: State,
    seat: Seat<State>,
}

impl Compositor {
    fn new() -> Self {
        let display = Display::<State>::new().unwrap();
        let dh = display.handle();
        let primary = PrimarySelectionState::new::<State>(&dh);
        let wlr = wlr_data_control::DataControlState::new::<State, _>(&dh, Some(&primary), |_| true);
        let ext = ext_data_control::DataControlState::new::<State, _>(&dh, Some(&primary), |_| true);
        let mut seat_state = SeatState::new();
        let seat = seat_state.new_wl_seat(&dh, "seat-0");
        let state = State {
            seat_state,
            data_device: DataDeviceState::new::<State>(&dh),
            primary,
            wlr,
            ext,
            reads: Vec::new(),
        };
        Self {
            display,
            dh,
            state,
            seat,
        }
    }

    /// Connect a client that binds the seat and every selection manager.
    fn connect(&mut self) -> (TestClient, Client) {
        let (server, socket) = UnixStream::pair().unwrap();
        let client = self.dh.insert_client(server, Arc::new(NoData)).unwrap();
        let conn = Connection::from_socket(socket).unwrap();
        let queue = conn.new_event_queue();
        let _registry = conn.display().get_registry(&queue.handle(), ());
        let mut test_client = TestClient {
            conn,
            queue,
            heard: Heard::default(),
        };
        self.pump(&mut [&mut test_client]);
        assert!(test_client.heard.seat.is_some());
        (test_client, client)
    }

    /// Let requests and events cross a few times.
    fn pump(&mut self, clients: &mut [&mut TestClient]) {
        for _ in 0..4 {
            for client in clients.iter_mut() {
                client.conn.flush().unwrap();
            }
            self.display.dispatch_clients(&mut self.state).unwrap();
            self.display.flush_clients().unwrap();
            for client in clients.iter_mut() {
                if let Some(guard) = client.queue.prepare_read() {
                    let _ = guard.read();
                }
                client.queue.dispatch_pending(&mut client.heard).unwrap();
            }
        }
    }

    /// Move the keyboard-following selection focus (both selections).
    fn focus(&mut self, client: Option<&Client>) {
        set_data_device_focus(&self.dh, &self.seat, client.cloned());
        set_primary_focus(&self.dh, &self.seat, client.cloned());
    }

    /// Offer a compositor-owned clipboard and primary selection of `mime_type`.
    fn select(&mut self, mime_type: &str) {
        set_data_device_selection(&self.dh, &self.seat, vec![mime_type.into()], ());
        set_primary_selection(&self.dh, &self.seat, vec![mime_type.into()], ());
    }
}

// The clients.

/// Which device heard a `selection` (or `primary_selection`) event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Device {
    Data,
    Primary,
    WlrClipboard,
    WlrPrimary,
    ExtClipboard,
    ExtPrimary,
}

#[derive(Debug, Clone)]
enum Offer {
    Data(WlDataOffer),
    Primary(ZwpPrimarySelectionOfferV1),
    Wlr(ZwlrDataControlOfferV1),
    Ext(ExtDataControlOfferV1),
}

impl Offer {
    fn receive(&self, mime_type: &str, fd: &OwnedFd) {
        let mime_type = mime_type.to_owned();
        match self {
            Offer::Data(offer) => offer.receive(mime_type, fd.as_fd()),
            Offer::Primary(offer) => offer.receive(mime_type, fd.as_fd()),
            Offer::Wlr(offer) => offer.receive(mime_type, fd.as_fd()),
            Offer::Ext(offer) => offer.receive(mime_type, fd.as_fd()),
        }
    }
}

#[derive(Default)]
struct Heard {
    seat: Option<WlSeat>,
    data_manager: Option<WlDataDeviceManager>,
    primary_manager: Option<ZwpPrimarySelectionDeviceManagerV1>,
    wlr_manager: Option<ZwlrDataControlManagerV1>,
    ext_manager: Option<ExtDataControlManagerV1>,
    /// Every `selection` event, in order: the offer, or `None` for an unset.
    selections: Vec<(Device, Option<Offer>)>,
    /// MIME types the client's data-control sources were asked to send.
    sends: Vec<String>,
}

struct TestClient {
    conn: Connection,
    queue: EventQueue<Heard>,
    heard: Heard,
}

impl TestClient {
    fn qh(&self) -> QueueHandle<Heard> {
        self.queue.handle()
    }

    /// Bind a `wl_data_device` and a primary-selection device.
    fn focus_devices(&self) -> (WlDataDevice, ZwpPrimarySelectionDeviceV1) {
        let seat = self.heard.seat.as_ref().unwrap();
        let data = self.heard.data_manager.as_ref().unwrap();
        let primary = self.heard.primary_manager.as_ref().unwrap();
        (
            data.get_data_device(seat, &self.qh(), ()),
            primary.get_device(seat, &self.qh(), ()),
        )
    }

    /// Bind a wlr and an ext data-control device.
    fn control_devices(&self) -> (ZwlrDataControlDeviceV1, ExtDataControlDeviceV1) {
        let seat = self.heard.seat.as_ref().unwrap();
        let wlr = self.heard.wlr_manager.as_ref().unwrap();
        let ext = self.heard.ext_manager.as_ref().unwrap();
        (
            wlr.get_data_device(seat, &self.qh(), ()),
            ext.get_data_device(seat, &self.qh(), ()),
        )
    }

    /// The last `selection` event `device` heard: `Some(None)` for an unset.
    fn last(&self, device: Device) -> Option<Option<Offer>> {
        self.heard
            .selections
            .iter()
            .rev()
            .find(|(d, _)| *d == device)
            .map(|(_, offer)| offer.clone())
    }

    /// The offer `device` holds now.
    fn offer(&self, device: Device) -> Offer {
        self.last(device)
            .flatten()
            .unwrap_or_else(|| panic!("{device:?} holds no offer"))
    }

    fn heard_count(&self, device: Device) -> usize {
        self.heard.selections.iter().filter(|(d, _)| *d == device).count()
    }
}

/// Ask `offer` for `mime_type` and return the read end of the pipe.
fn receive(offer: &Offer, mime_type: &str) -> OwnedFd {
    let (read, write) = pipe_with(PipeFlags::CLOEXEC | PipeFlags::NONBLOCK).unwrap();
    offer.receive(mime_type, &write);
    read
}

/// Everything in the pipe once every writer closed it; `None` while one
/// still holds it open.
fn drain(read: &OwnedFd) -> Option<Vec<u8>> {
    let mut data = Vec::new();
    let mut buf = [0u8; 256];
    loop {
        match rustix::io::read(read, &mut buf) {
            Ok(0) => return Some(data),
            Ok(n) => data.extend_from_slice(&buf[..n]),
            Err(rustix::io::Errno::AGAIN) => return None,
            Err(err) => panic!("read: {err}"),
        }
    }
}

impl Dispatch<WlRegistry, ()> for Heard {
    fn event(
        heard: &mut Self,
        registry: &WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_seat" => heard.seat = Some(registry.bind(name, version.min(7), qh, ())),
                "wl_data_device_manager" => {
                    heard.data_manager = Some(registry.bind(name, version.min(3), qh, ()))
                }
                "zwp_primary_selection_device_manager_v1" => {
                    heard.primary_manager = Some(registry.bind(name, 1, qh, ()))
                }
                "zwlr_data_control_manager_v1" => {
                    heard.wlr_manager = Some(registry.bind(name, version.min(2), qh, ()))
                }
                "ext_data_control_manager_v1" => heard.ext_manager = Some(registry.bind(name, 1, qh, ())),
                _ => (),
            }
        }
    }
}

delegate_noop!(Heard: ignore WlSeat);
delegate_noop!(Heard: WlDataDeviceManager);
delegate_noop!(Heard: ZwpPrimarySelectionDeviceManagerV1);
delegate_noop!(Heard: ZwlrDataControlManagerV1);
delegate_noop!(Heard: ExtDataControlManagerV1);
delegate_noop!(Heard: ignore WlDataOffer);
delegate_noop!(Heard: ignore ZwpPrimarySelectionOfferV1);
delegate_noop!(Heard: ignore ZwlrDataControlOfferV1);
delegate_noop!(Heard: ignore ExtDataControlOfferV1);

impl Dispatch<WlDataDevice, ()> for Heard {
    fn event(
        heard: &mut Self,
        _: &WlDataDevice,
        event: wl_data_device::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_data_device::Event::Selection { id } = event {
            heard.selections.push((Device::Data, id.map(Offer::Data)));
        }
    }

    event_created_child!(Heard, WlDataDevice, [
        wl_data_device::EVT_DATA_OFFER_OPCODE => (WlDataOffer, ()),
    ]);
}

impl Dispatch<ZwpPrimarySelectionDeviceV1, ()> for Heard {
    fn event(
        heard: &mut Self,
        _: &ZwpPrimarySelectionDeviceV1,
        event: zwp_primary_selection_device_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwp_primary_selection_device_v1::Event::Selection { id } = event {
            heard.selections.push((Device::Primary, id.map(Offer::Primary)));
        }
    }

    event_created_child!(Heard, ZwpPrimarySelectionDeviceV1, [
        zwp_primary_selection_device_v1::EVT_DATA_OFFER_OPCODE => (ZwpPrimarySelectionOfferV1, ()),
    ]);
}

impl Dispatch<ZwlrDataControlDeviceV1, ()> for Heard {
    fn event(
        heard: &mut Self,
        _: &ZwlrDataControlDeviceV1,
        event: zwlr_data_control_device_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwlr_data_control_device_v1::Event::Selection { id } => {
                heard.selections.push((Device::WlrClipboard, id.map(Offer::Wlr)))
            }
            zwlr_data_control_device_v1::Event::PrimarySelection { id } => {
                heard.selections.push((Device::WlrPrimary, id.map(Offer::Wlr)))
            }
            _ => (),
        }
    }

    event_created_child!(Heard, ZwlrDataControlDeviceV1, [
        zwlr_data_control_device_v1::EVT_DATA_OFFER_OPCODE => (ZwlrDataControlOfferV1, ()),
    ]);
}

impl Dispatch<ExtDataControlDeviceV1, ()> for Heard {
    fn event(
        heard: &mut Self,
        _: &ExtDataControlDeviceV1,
        event: ext_data_control_device_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            ext_data_control_device_v1::Event::Selection { id } => {
                heard.selections.push((Device::ExtClipboard, id.map(Offer::Ext)))
            }
            ext_data_control_device_v1::Event::PrimarySelection { id } => {
                heard.selections.push((Device::ExtPrimary, id.map(Offer::Ext)))
            }
            _ => (),
        }
    }

    event_created_child!(Heard, ExtDataControlDeviceV1, [
        ext_data_control_device_v1::EVT_DATA_OFFER_OPCODE => (ExtDataControlOfferV1, ()),
    ]);
}

impl Dispatch<ZwlrDataControlSourceV1, ()> for Heard {
    fn event(
        heard: &mut Self,
        _: &ZwlrDataControlSourceV1,
        event: zwlr_data_control_source_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwlr_data_control_source_v1::Event::Send { mime_type, fd } = event {
            rustix::io::write(&fd, mime_type.as_bytes()).unwrap();
            heard.sends.push(mime_type);
        }
    }
}

// The cases.

const TEXT: &str = "text/plain;charset=utf-8";
const HTML: &str = "text/html";

#[test]
fn focus_loss_unsets_the_old_clients_selections() {
    let mut comp = Compositor::new();
    let (mut a, a_id) = comp.connect();
    let (mut b, b_id) = comp.connect();
    let _a_devices = a.focus_devices();
    let _b_devices = b.focus_devices();
    comp.pump(&mut [&mut a, &mut b]);

    comp.select(TEXT);
    comp.focus(Some(&a_id));
    comp.pump(&mut [&mut a, &mut b]);
    assert!(matches!(a.last(Device::Data), Some(Some(Offer::Data(_)))));
    assert!(matches!(a.last(Device::Primary), Some(Some(Offer::Primary(_)))));
    assert!(b.last(Device::Data).is_none() && b.last(Device::Primary).is_none());

    comp.focus(Some(&b_id));
    comp.pump(&mut [&mut a, &mut b]);
    assert!(matches!(a.last(Device::Data), Some(None)), "wl_data_device unset");
    assert!(matches!(a.last(Device::Primary), Some(None)), "primary unset");
    assert!(matches!(b.last(Device::Data), Some(Some(Offer::Data(_)))));
    assert!(matches!(b.last(Device::Primary), Some(Some(Offer::Primary(_)))));

    // No focus at all: the last holder is told too.
    comp.focus(None);
    comp.pump(&mut [&mut a, &mut b]);
    assert!(matches!(b.last(Device::Data), Some(None)));
    assert!(matches!(b.last(Device::Primary), Some(None)));
}

#[test]
fn a_stale_offer_or_one_whose_client_lost_focus_is_refused() {
    let mut comp = Compositor::new();
    let (mut a, a_id) = comp.connect();
    let (mut b, b_id) = comp.connect();
    let _a_devices = a.focus_devices();
    let _b_devices = b.focus_devices();
    comp.pump(&mut [&mut a, &mut b]);

    comp.select(TEXT);
    comp.focus(Some(&a_id));
    comp.pump(&mut [&mut a, &mut b]);
    let first = a.offer(Device::Data);
    let first_primary = a.offer(Device::Primary);
    let pipe = receive(&first, TEXT);
    comp.pump(&mut [&mut a, &mut b]);
    assert_eq!(
        drain(&pipe).as_deref(),
        Some(TEXT.as_bytes()),
        "the current offer is read"
    );

    // A newer selection while A keeps the focus: the old offer is stale.
    comp.select(HTML);
    comp.pump(&mut [&mut a, &mut b]);
    comp.state.reads.clear();
    let stale = receive(&first, TEXT);
    let stale_primary = receive(&first_primary, TEXT);
    comp.pump(&mut [&mut a, &mut b]);
    assert_eq!(drain(&stale).as_deref(), Some(&[][..]), "end of file, no data");
    assert_eq!(drain(&stale_primary).as_deref(), Some(&[][..]));
    assert!(comp.state.reads.is_empty());
    let second = a.offer(Device::Data);
    let second_primary = a.offer(Device::Primary);
    let pipe = receive(&second, HTML);
    comp.pump(&mut [&mut a, &mut b]);
    assert_eq!(drain(&pipe).as_deref(), Some(HTML.as_bytes()));

    // Focus moves to B: A's newest offers are refused, B's are read.
    comp.focus(Some(&b_id));
    comp.pump(&mut [&mut a, &mut b]);
    comp.state.reads.clear();
    let unfocused = receive(&second, HTML);
    let unfocused_primary = receive(&second_primary, HTML);
    comp.pump(&mut [&mut a, &mut b]);
    assert_eq!(drain(&unfocused).as_deref(), Some(&[][..]));
    assert_eq!(drain(&unfocused_primary).as_deref(), Some(&[][..]));
    assert!(
        comp.state.reads.is_empty(),
        "nothing reached the selection's owner"
    );
    let pipe = receive(&b.offer(Device::Primary), HTML);
    comp.pump(&mut [&mut a, &mut b]);
    assert_eq!(drain(&pipe).as_deref(), Some(HTML.as_bytes()));
    assert_eq!(comp.state.reads, [(SelectionTarget::Primary, HTML.to_owned())]);

    // Back to A: the offer it held before stays stale; the new one is read.
    comp.focus(Some(&a_id));
    comp.pump(&mut [&mut a, &mut b]);
    comp.state.reads.clear();
    let old = receive(&second, HTML);
    comp.pump(&mut [&mut a, &mut b]);
    assert_eq!(drain(&old).as_deref(), Some(&[][..]));
    assert!(comp.state.reads.is_empty());
    let pipe = receive(&a.offer(Device::Data), HTML);
    comp.pump(&mut [&mut a, &mut b]);
    assert_eq!(drain(&pipe).as_deref(), Some(HTML.as_bytes()));
}

#[test]
fn a_client_owned_selection_is_not_sent_for_a_stale_offer() {
    let mut comp = Compositor::new();
    let (mut a, a_id) = comp.connect();
    let (mut b, b_id) = comp.connect();
    let (mut manager, _) = comp.connect();
    let _a_devices = a.focus_devices();
    let _b_devices = b.focus_devices();
    let (wlr, _ext) = manager.control_devices();
    comp.focus(Some(&a_id));
    comp.pump(&mut [&mut a, &mut b, &mut manager]);

    // A clipboard manager owns the selection; A holds an offer of it.
    let source = manager
        .heard
        .wlr_manager
        .as_ref()
        .unwrap()
        .create_data_source(&manager.qh(), ());
    source.offer(TEXT.into());
    wlr.set_selection(Some(&source));
    comp.pump(&mut [&mut a, &mut b, &mut manager]);
    let held = a.offer(Device::Data);

    comp.focus(Some(&b_id));
    comp.pump(&mut [&mut a, &mut b, &mut manager]);
    let refused = receive(&held, TEXT);
    comp.pump(&mut [&mut a, &mut b, &mut manager]);
    assert!(manager.heard.sends.is_empty(), "the owner was not asked");
    assert_eq!(drain(&refused).as_deref(), Some(&[][..]));

    let pipe = receive(&b.offer(Device::Data), TEXT);
    comp.pump(&mut [&mut a, &mut b, &mut manager]);
    assert_eq!(manager.heard.sends, [TEXT]);
    assert_eq!(drain(&pipe).as_deref(), Some(TEXT.as_bytes()));
}

#[test]
fn data_control_devices_do_not_follow_focus() {
    let mut comp = Compositor::new();
    let (mut a, a_id) = comp.connect();
    let (mut b, b_id) = comp.connect();
    let (mut manager, _) = comp.connect();
    let _a_devices = a.focus_devices();
    let _b_devices = b.focus_devices();
    let _manager_devices = manager.control_devices();
    comp.pump(&mut [&mut a, &mut b, &mut manager]);

    comp.focus(Some(&a_id));
    comp.select(TEXT);
    comp.pump(&mut [&mut a, &mut b, &mut manager]);
    let devices = [
        Device::WlrClipboard,
        Device::WlrPrimary,
        Device::ExtClipboard,
        Device::ExtPrimary,
    ];
    let offers = devices.map(|device| manager.offer(device));
    let counts = devices.map(|device| manager.heard_count(device));

    // Focus changes: the manager hears nothing and its offers stay readable.
    comp.focus(Some(&b_id));
    comp.focus(None);
    comp.focus(Some(&a_id));
    comp.pump(&mut [&mut a, &mut b, &mut manager]);
    assert_eq!(devices.map(|device| manager.heard_count(device)), counts);
    comp.state.reads.clear();
    let pipes = offers.each_ref().map(|offer| receive(offer, TEXT));
    comp.pump(&mut [&mut a, &mut b, &mut manager]);
    for (device, pipe) in devices.iter().zip(&pipes) {
        assert_eq!(drain(pipe).as_deref(), Some(TEXT.as_bytes()), "{device:?}");
    }
    assert_eq!(comp.state.reads.len(), 4);
}
