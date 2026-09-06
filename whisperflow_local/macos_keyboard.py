"""Darwin fixes around pynput's single, persistent keyboard event tap.

pynput 1.8.2 treats tap-disabled notifications as keys and decodes modifier
releases using a flag shared by both sides. Keep its layout/key translation,
but handle those two cases before passing events to the stock listener.
"""
import Quartz
from pynput import keyboard


# Virtual key -> (this side, either side, family), from IOKit/IOLLEvent.h's
# NX_DEVICE*KEYMASK constants. The public CGEvent modifier flags combine sides.
_MODIFIER_MASKS = {
    56: (0x0002, 0x0006, Quartz.kCGEventFlagMaskShift),  # left Shift
    60: (0x0004, 0x0006, Quartz.kCGEventFlagMaskShift),  # right Shift
    59: (0x0001, 0x2001, Quartz.kCGEventFlagMaskControl),  # left Control
    62: (0x2000, 0x2001, Quartz.kCGEventFlagMaskControl),  # right Control
    58: (0x0020, 0x0060, Quartz.kCGEventFlagMaskAlternate),  # left Option
    61: (0x0040, 0x0060, Quartz.kCGEventFlagMaskAlternate),  # right Option
    55: (0x0008, 0x0018, Quartz.kCGEventFlagMaskCommand),  # left Command
    54: (0x0010, 0x0018, Quartz.kCGEventFlagMaskCommand),  # right Command
}
_TAP_DISABLED = {
    Quartz.kCGEventTapDisabledByTimeout,
    Quartz.kCGEventTapDisabledByUserInput,
}


class MacOSKeyboardListener(keyboard.Listener):
    def __init__(self, *args, on_tap_recovered=None, **kwargs):
        self._event_tap = None
        self._on_tap_recovered = on_tap_recovered
        super().__init__(*args, **kwargs)

    def _create_event_tap(self):
        self._event_tap = super()._create_event_tap()
        return self._event_tap

    def _recover_tap(self, reason):
        if self._event_tap is not None:
            Quartz.CGEventTapEnable(self._event_tap, True)
            if self._on_tap_recovered is not None:
                self._on_tap_recovered(reason)

    def _handler(self, proxy, event_type, event, refcon):
        if event_type in _TAP_DISABLED:
            # These are control notifications, potentially with a NULL event.
            # Never let pynput read a keycode or injected PID from them.
            self._recover_tap("timeout" if event_type ==
                              Quartz.kCGEventTapDisabledByTimeout else "user input")
            return event
        return super()._handler(proxy, event_type, event, refcon)

    def check_health(self):
        """Also recover if the tap was disabled without a delivered callback."""
        if (self.running and self._event_tap is not None
                and not Quartz.CGEventTapIsEnabled(self._event_tap)):
            self._recover_tap("health check")

    def _handle_message(self, proxy, event_type, event, refcon, injected):
        if event_type == Quartz.kCGEventFlagsChanged:
            vk = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode)
            masks = _MODIFIER_MASKS.get(vk)
            if masks is not None:
                flags = Quartz.CGEventGetFlags(event)
                key = self._event_to_key(event)
                side, either, family = masks
                # Some synthesized events only supply the aggregate flag.
                # Use that fallback only when neither side bit is present.
                pressed = bool(flags & side) if flags & either else bool(
                    flags & family)
                self._flags = flags
                (self.on_press if pressed else self.on_release)(key, injected)
                return
        return super()._handle_message(proxy, event_type, event, refcon, injected)

    @staticmethod
    def key_is_pressed(key):
        """Read the session's current key state, independently of tap delivery.

        Modifiers use flags state: their flagsChanged events need not update
        the ordinary key bitmap. CGEventSourceKeyState can therefore report
        false for Right Option throughout a real physical hold.

        Unknown/media keys and modifier flags without side information return
        None so an ambiguous state never ends a valid hold.
        """
        code = key.value if isinstance(key, keyboard.Key) else key
        vk = getattr(code, "vk", None)
        if vk is None or getattr(code, "_is_media", False):
            return None
        masks = _MODIFIER_MASKS.get(vk)
        if masks is not None:
            flags = Quartz.CGEventSourceFlagsState(
                Quartz.kCGEventSourceStateCombinedSessionState)
            side, either, family = masks
            if flags & either:
                return bool(flags & side)
            # An aggregate flag alone cannot tell which side is held.
            return None if flags & family else False
        return bool(Quartz.CGEventSourceKeyState(
            Quartz.kCGEventSourceStateCombinedSessionState, vk))
