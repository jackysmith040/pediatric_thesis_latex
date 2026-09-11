from nicegui import ui
from src.ui.components import page_container, primary_button

def register_landing():
    @ui.page('/')
    def landing():
        with page_container(scrollable=True):
            # Minimalist Hero Section
            with ui.column().classes('w-full max-w-[1200px] mx-auto mt-24 px-4 md:px-12 flex-col items-center justify-center min-h-[50vh]'):
                # Premium Badging
                with ui.row().classes('items-center gap-3 bg-slate-900/80 border border-slate-800 rounded-full px-4 py-2 mb-8 shadow-lg backdrop-blur-sm'):
                    ui.element('div').classes('w-2 h-2 rounded-full bg-emerald-500 animate-pulse')
                    ui.label('SYSTEM ONLINE - V2.0.0').classes('text-xs font-bold tracking-[0.2em] text-slate-300')

                ui.label('Pediatric Monitor').classes('text-5xl md:text-8xl font-black text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400 tracking-tighter mb-6 drop-shadow-2xl text-center')
                
                ui.label('A centralized telemetry platform utilizing localized computer vision to dynamically track wait room capacities. No external APIs, no latency, zero pediatric overcrowding.').classes('text-lg md:text-2xl text-slate-400 max-w-4xl text-center leading-relaxed font-light mb-12')
                
                with ui.row().classes('gap-6 justify-center w-full'):
                    primary_button('Initialize Triage Monitor', on_click=lambda: ui.navigate.to('/dashboard'))
                    
            # Data Features Grid
            with ui.row().classes('w-full max-w-[1400px] mx-auto mt-16 px-4 md:px-12 gap-8 grid grid-cols-1 md:grid-cols-3 pb-32'):
                
                with ui.card().classes('w-full p-8 md:p-10 bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800/60 rounded-[2rem] shadow-2xl hover:shadow-indigo-500/10 hover:border-indigo-500/30 transition-all duration-300 group'):
                    with ui.row().classes('w-16 h-16 rounded-2xl bg-indigo-500/10 items-center justify-center mb-6 group-hover:scale-110 transition-transform'):
                        ui.icon('health_and_safety').classes('text-3xl text-indigo-400')
                    ui.label('Zero-Latency').classes('text-2xl font-bold text-white mb-3 tracking-tight')
                    ui.label('Inference runs in parallel threads on the native device to guarantee completely uninterrupted 30fps video streams.').classes('text-slate-400 text-base leading-relaxed')
                    
                with ui.card().classes('w-full p-8 md:p-10 bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800/60 rounded-[2rem] shadow-2xl hover:shadow-indigo-500/10 hover:border-indigo-500/30 transition-all duration-300 group'):
                    with ui.row().classes('w-16 h-16 rounded-2xl bg-indigo-500/10 items-center justify-center mb-6 group-hover:scale-110 transition-transform'):
                        ui.icon('analytics').classes('text-3xl text-indigo-400')
                    ui.label('Debounced Telemetry').classes('text-2xl font-bold text-white mb-3 tracking-tight')
                    ui.label('The monolithic Python stack actively merges ID fragmentation via spatial bounding-box checks for pure accuracy.').classes('text-slate-400 text-base leading-relaxed')
                    
                with ui.card().classes('w-full p-8 md:p-10 bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800/60 rounded-[2rem] shadow-2xl hover:shadow-indigo-500/10 hover:border-indigo-500/30 transition-all duration-300 group'):
                    with ui.row().classes('w-16 h-16 rounded-2xl bg-indigo-500/10 items-center justify-center mb-6 group-hover:scale-110 transition-transform'):
                        ui.icon('security').classes('text-3xl text-indigo-400')
                    ui.label('Privacy By Design').classes('text-2xl font-bold text-white mb-3 tracking-tight')
                    ui.label('All YOLOv26 tensor evaluations occur entirely on-device. No visual data is ever transmitted to the cloud.').classes('text-slate-400 text-base leading-relaxed')
