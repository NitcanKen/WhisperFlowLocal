"""Frontmost-application detection via AppKit (NSWorkspace)."""


def frontmost_app_name() -> str:
    """Return the localized name of the frontmost app, or '' if unknown."""
    try:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return ""
        return str(app.localizedName() or "")
    except Exception:
        return ""
