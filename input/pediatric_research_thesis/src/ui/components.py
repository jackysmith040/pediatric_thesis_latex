from nicegui import ui
from contextlib import contextmanager

@contextmanager
def page_container(scrollable: bool = False):
    """A standard full-screen container with the dark mode background."""
    ui.colors(primary='#4f46e5') # Indigo-600
    ui.query('html, body, .nicegui-content').classes('p-0 m-0 w-full h-full min-h-screen bg-slate-950')
    overflow_class = 'overflow-x-hidden min-h-screen' if scrollable else 'overflow-hidden h-screen'
    with ui.column().classes(f'w-full {overflow_class} bg-slate-950 p-0 m-0 text-slate-100 gap-0 flex-nowrap') as container:
        yield container


@contextmanager
def dashboard_card(max_width='max-w-2xl'):
    """A reusable bounded card with dark mode styling."""
    with ui.card().classes(f'w-full {max_width} p-8 md:p-12 items-center text-center shadow-2xl shadow-black/50 rounded-[2rem] bg-slate-900 border border-slate-800') as card:
        yield card

def primary_button(text: str, on_click=None):
    """Primary action button."""
    return ui.button(text, on_click=on_click) \
        .props('unelevated rounded size=lg') \
        .classes('w-full sm:w-auto px-8 font-bold shadow-lg shadow-indigo-900/30 bg-indigo-600 hover:bg-indigo-500 text-white transition-transform hover:scale-105')

def secondary_button(text: str, on_click=None):
    """Secondary action button."""
    return ui.button(text, on_click=on_click) \
        .props('outline rounded size=lg') \
        .classes('w-full sm:w-auto px-8 font-bold border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white transition-colors')

def hero_icon(icon_name: str):
    """Standardized icon for card headers."""
    return ui.icon(icon_name).classes('text-6xl text-indigo-400 mb-4 drop-shadow-sm')

def hero_title(text: str):
    """Standardized main title."""
    return ui.label(text).classes('text-4xl md:text-5xl font-extrabold mb-4 text-white tracking-tight')

def hero_subtitle(text: str):
    """Standardized subtitle."""
    return ui.label(text).classes('text-lg md:text-xl text-slate-400 mb-10 max-w-lg leading-relaxed')

@contextmanager
def navbar(title: str):
    with ui.row().classes('w-full items-center justify-between px-8 py-4 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 sticky top-0 z-50'):
        with ui.row().classes('items-center gap-4'):
            ui.button(on_click=lambda: ui.navigate.to('/')).props('flat round icon=arrow_back').classes('text-slate-400 hover:text-white hover:bg-slate-800')
            ui.label(title).classes('text-xl font-bold tracking-tight text-white')
        
        with ui.row().classes('items-center gap-2'):
            ui.element('div').classes('w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_10px_#10b981]')
            ui.label('System Active').classes('text-xs font-semibold tracking-wider text-emerald-500 uppercase')
        yield

def alert_banner(title: str, message: str, obj, attr: str):
    with ui.row().classes('w-full bg-red-500/10 border border-red-500/30 rounded-2xl p-5 items-start gap-4 text-red-400 shadow-[0_0_20px_rgba(239,68,68,0.15)]').bind_visibility_from(obj, attr):
        ui.icon('warning').classes('text-2xl mt-1')
        with ui.column().classes('gap-1'):
            ui.label(title).classes('font-bold tracking-wide uppercase text-sm')
            ui.label(message).classes('text-sm opacity-90 leading-relaxed')

def stat_card(title: str, obj, attr: str, is_primary: bool = True):
    with ui.card().classes('w-full p-6 rounded-2xl bg-slate-800/80 border border-slate-700 hover:bg-slate-700 transition-colors relative overflow-hidden group'):
        if is_primary:
            ui.element('div').classes('absolute inset-0 bg-gradient-to-br from-indigo-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity')
        ui.label(title).classes('text-xs font-semibold text-slate-400 tracking-[0.1em] uppercase mb-2')
        color_class = 'text-white drop-shadow-[0_0_15px_rgba(255,255,255,0.2)]' if is_primary else 'text-slate-300'
        ui.label().bind_text_from(obj, attr).classes(f'text-6xl font-extrabold tracking-tighter {color_class} relative z-10')

@contextmanager
def video_feed_card(title: str):
    with ui.card().classes('w-full h-full min-h-[500px] p-0 rounded-3xl bg-slate-900 border border-slate-800 overflow-hidden flex flex-col shadow-2xl relative'):
        # Header (Floating overlay over top of video feed)
        with ui.row().classes('w-full px-6 py-4 border-b border-slate-800/80 bg-slate-900/70 backdrop-blur-md justify-between items-center z-20 absolute top-0 left-0 right-0'):
            ui.label(title).classes('text-sm font-semibold text-slate-200 uppercase tracking-[0.1em]')
            ui.label('Monolith Engine').classes('text-xs font-mono text-indigo-400 bg-indigo-500/10 px-2 py-1 rounded-full border border-indigo-500/20')
        # Body wrapper - flex-1 with min-h ensures proper Flexbox layout calculation instantly on page load
        with ui.element('div').classes('w-full flex-1 relative bg-slate-950 min-h-[450px] overflow-hidden'):
            yield
