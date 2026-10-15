use wayland_protocols_wlr::data_control::v1::server::zwlr_data_control_device_v1::EVT_PRIMARY_SELECTION_SINCE;
use wayland_server::backend::ClientId;
use wayland_server::protocol::wl_data_device::WlDataDevice;
use wayland_server::{Client, DisplayHandle, Resource};

use super::device::SelectionDevice;
use super::offer::{OfferReplySource, SelectionOffer};
use super::{SelectionHandler, SelectionTarget};

/// Seat data used to handle regular selection operations.
///
/// The data is shared across primary, data device, and data control selections.
pub(crate) struct SeatData<U: Clone + Sync + Send + 'static> {
    known_devices: Vec<SelectionDevice>,
    clipboard_selection: Option<OfferReplySource<U>>,
    clipboard_selection_focus: Option<Client>,
    primary_selection: Option<OfferReplySource<U>>,
    primary_selection_focus: Option<Client>,
    /// Advanced whenever the clipboard selection or its focus changes: a
    /// `wl_data_offer` made at an older generation is no longer readable.
    clipboard_generation: u64,
    /// The same for the primary selection.
    primary_generation: u64,
}

impl<U: Clone + Send + Sync + 'static> SeatData<U> {
    /// Create a new [`SeatData`] with empty selections and without focusing any client.
    pub fn new() -> Self {
        Default::default()
    }

    pub fn retain_devices<F: FnMut(&SelectionDevice) -> bool>(&mut self, retain: F) {
        self.known_devices.retain(retain);
    }

    /// Register a new [SelectionDevice] into [SeatData].
    pub fn add_device(&mut self, device: SelectionDevice) {
        self.known_devices.push(device);
    }

    /// Get the iterator over [`WlDataDevice`]. Mostly used for the drag and drop handling.
    pub fn known_data_devices(&self) -> impl Iterator<Item = &WlDataDevice> {
        self.known_devices.iter().filter_map(|device| match device {
            SelectionDevice::DataDevice(device) => Some(device),
            _ => None,
        })
    }

    /// Change focus for the clipboard selection to `new_focus` client. Providing `None` will
    /// remove the focus.
    ///
    /// If the focus is different from the existing focus, the current selection will be offered to the client.
    pub fn set_clipboard_focus<D>(&mut self, dh: &DisplayHandle, new_focus: Option<Client>)
    where
        D: SelectionHandler<SelectionUserData = U> + 'static,
    {
        if self.clipboard_selection_focus != new_focus {
            let old_focus = std::mem::replace(&mut self.clipboard_selection_focus, new_focus);
            self.clipboard_generation += 1;
            self.unset_for(dh, SelectionTarget::Clipboard, old_focus.as_ref());
            self.send_selection::<D>(dh, SelectionTarget::Clipboard, None, false)
        }
    }

    /// Set the clipboard selection to the `new_selection`, providing `None` will clear the
    /// selection.
    pub fn set_clipboard_selection<D>(
        &mut self,
        dh: &DisplayHandle,
        new_selection: Option<OfferReplySource<U>>,
    ) where
        D: SelectionHandler<SelectionUserData = U> + 'static,
    {
        if let Some(OfferReplySource::Client(source)) = self.clipboard_selection.as_ref() {
            match new_selection.as_ref() {
                Some(OfferReplySource::Client(new_source)) if new_source == source => (),
                _ => source.cancel(),
            }
        }
        self.clipboard_selection = new_selection;
        self.clipboard_generation += 1;
        self.send_selection::<D>(dh, SelectionTarget::Clipboard, None, true)
    }

    /// Change focus for the primary selection to `new_focus` client. Providing `None` will
    /// remove the focus.
    ///
    /// If the focus is different from the existing focus, the current selection will be offered to the client.
    pub fn set_primary_focus<D>(&mut self, dh: &DisplayHandle, new_focus: Option<Client>)
    where
        D: SelectionHandler<SelectionUserData = U> + 'static,
    {
        if self.primary_selection_focus != new_focus {
            let old_focus = std::mem::replace(&mut self.primary_selection_focus, new_focus);
            self.primary_generation += 1;
            self.unset_for(dh, SelectionTarget::Primary, old_focus.as_ref());
            self.send_selection::<D>(dh, SelectionTarget::Primary, None, false)
        }
    }

    /// Set the primary selection to the `new_selection`, providing `None` will clear the
    /// selection.
    pub fn set_primary_selection<D>(&mut self, dh: &DisplayHandle, new_selection: Option<OfferReplySource<U>>)
    where
        D: SelectionHandler<SelectionUserData = U> + 'static,
    {
        if let Some(OfferReplySource::Client(source)) = self.primary_selection.as_ref() {
            match new_selection.as_ref() {
                Some(OfferReplySource::Client(new_source)) if new_source == source => (),
                _ => source.cancel(),
            }
        }
        self.primary_selection = new_selection;
        self.primary_generation += 1;
        self.send_selection::<D>(dh, SelectionTarget::Primary, None, true)
    }

    /// Tell `client`, which just lost the focus for `ty`, that it no longer has a
    /// selection: its offer stopped being valid with the focus (the protocol says
    /// so), and its devices hear it instead of keeping a stale offer around.
    fn unset_for(&self, dh: &DisplayHandle, ty: SelectionTarget, client: Option<&Client>) {
        let Some(client) = client else {
            return;
        };
        for device in &self.known_devices {
            let owned = dh.get_client(device.id()).is_ok_and(|c| &c == client);
            match device {
                SelectionDevice::DataDevice(_) if owned && ty == SelectionTarget::Clipboard => {
                    device.unset_selection()
                }
                SelectionDevice::Primary(_) if owned && ty == SelectionTarget::Primary => {
                    device.unset_primary_selection()
                }
                _ => (),
            }
        }
    }

    /// Whether a focus-following offer made at `generation` for `ty` may still be
    /// read by `client`: no selection or focus change since, and `client` holds
    /// the focus. Data-control offers are not gated.
    pub fn offer_is_current(&self, ty: SelectionTarget, generation: u64, client: &ClientId) -> bool {
        let (current, focus) = match ty {
            SelectionTarget::Clipboard => (self.clipboard_generation, &self.clipboard_selection_focus),
            SelectionTarget::Primary => (self.primary_generation, &self.primary_selection_focus),
        };
        current == generation && focus.as_ref().is_some_and(|focus| &focus.id() == client)
    }

    /// Get the current selection which will be used as a clipboard selection.
    pub fn get_clipboard_selection(&self) -> Option<&OfferReplySource<U>> {
        self.clipboard_selection.as_ref()
    }

    /// Get the current selection which will be used as a primary selection.
    pub fn get_primary_selection(&self) -> Option<&OfferReplySource<U>> {
        self.primary_selection.as_ref()
    }

    /// Send selection for the given [`SelectionTarget`]. The data control devices should only
    /// be update when the new selection is actually being set.
    ///
    /// The `restrict_to` option ensures that the selection is only send to the specified device,
    /// however, when the previous selection source provider dies. It'll re-broadcast the state.
    ///
    /// `update_data_control` checks whether the data control devices should be updated. Usually
    /// they shouldn't be updated when you just change the focus between clients, however, when
    /// selection source dies the state is re-broadcasted.
    pub fn send_selection<D>(
        &mut self,
        dh: &DisplayHandle,
        ty: SelectionTarget,
        restrict_to: Option<&SelectionDevice>,
        update_data_control: bool,
    ) where
        D: SelectionHandler<SelectionUserData = U> + 'static,
    {
        let (client, selection, generation) = match ty {
            SelectionTarget::Primary => (
                self.primary_selection_focus.as_ref(),
                &mut self.primary_selection,
                self.primary_generation,
            ),
            SelectionTarget::Clipboard => (
                self.clipboard_selection_focus.as_ref(),
                &mut self.clipboard_selection,
                self.clipboard_generation,
            ),
        };

        for device in self
            .known_devices
            .iter()
            .filter(|&device| restrict_to.is_none_or(|r| r == device))
            .filter(|&device| match device {
                // NOTE: filter by actual type here to not get a missmpatches when using selections
                // later on.
                SelectionDevice::DataDevice(_) => ty == SelectionTarget::Clipboard,
                SelectionDevice::Primary(_) => ty == SelectionTarget::Primary,
                SelectionDevice::WlrDataControl(data_control) => {
                    // Primary selection is available for data control only since v2.
                    update_data_control
                        && (data_control.version() >= EVT_PRIMARY_SELECTION_SINCE
                            || ty != SelectionTarget::Primary)
                }
                SelectionDevice::ExtDataControl(_) => update_data_control,
            })
        {
            // Data control doesn't require focus and should always get selection updates, unless
            // it was requested not to update them.
            if !matches!(
                device,
                SelectionDevice::WlrDataControl(_) | SelectionDevice::ExtDataControl(_)
            ) && dh
                .get_client(device.id())
                .map(|c| Some(&c) != client)
                .unwrap_or(true)
            {
                continue;
            }

            match (&*selection, ty) {
                (None, SelectionTarget::Primary) => {
                    device.unset_primary_selection();
                    continue;
                }
                (None, SelectionTarget::Clipboard) => {
                    device.unset_selection();
                    continue;
                }
                (Some(selection), _) => {
                    // DataControl devices is the client itself, however other devices use
                    // the currently focused one as a client.
                    let client_id = match device {
                        SelectionDevice::WlrDataControl(device) => {
                            dh.get_client(device.id()).ok().map(|c| c.id())
                        }
                        SelectionDevice::ExtDataControl(device) => {
                            dh.get_client(device.id()).ok().map(|c| c.id())
                        }
                        _ => client.map(|c| c.id()),
                    };

                    let client_id = match client_id {
                        Some(client_id) => client_id,
                        None => continue,
                    };

                    // Offers that follow keyboard focus are readable only while they
                    // are current; data-control offers are not gated.
                    let gate = match device {
                        SelectionDevice::WlrDataControl(_) | SelectionDevice::ExtDataControl(_) => None,
                        _ => Some((ty, generation)),
                    };
                    let offer = SelectionOffer::new::<D>(dh, device, client_id, selection.clone(), gate);

                    device.offer(&offer);

                    for mime_type in selection.mime_types() {
                        offer.offer(mime_type);
                    }

                    match ty {
                        SelectionTarget::Primary => device.primary_selection(&offer),
                        SelectionTarget::Clipboard => device.selection(&offer),
                    }
                }
            };
        }
    }
}

impl<U: Clone + Send + Sync + 'static> Default for SeatData<U> {
    fn default() -> Self {
        Self {
            known_devices: Vec::new(),
            clipboard_selection: None,
            clipboard_selection_focus: None,
            primary_selection: None,
            primary_selection_focus: None,
            clipboard_generation: 0,
            primary_generation: 0,
        }
    }
}
