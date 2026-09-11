from nicegui import ui, app, run
import asyncio
from src.ui.components import (
    page_container, 
    navbar,
    alert_banner,
    stat_card,
    video_feed_card
)
from src.state.telemetry import TelemetryState
from src.engine.detector import Detector
from src.engine.stream_resolver import PRESET_TEST_STREAMS

from typing import Callable, Optional

def register_dashboard(state: TelemetryState, get_detector: Callable[[], Optional[Detector]]):
    @ui.page('/dashboard')
    def dashboard():
        # Update progress bars manually since linear_progress value isn't auto-bound nicely from models
        def update_progress():
            try:
                total_p = max(1, state.total_daily_adults + state.total_daily_children)
                p_child.value = state.total_daily_children / total_p
                p_adult.value = state.total_daily_adults / total_p
            except Exception:
                pass

        ui.timer(1.0, update_progress)

        with page_container():
            with navbar('Clinical Command Center'):
                pass
            
            # The core layout: flex-grow fills screen, overflow-y-auto allows vertical scrolling if needed
            with ui.row().classes('w-full h-full flex-grow p-4 md:p-6 gap-6 items-stretch overflow-y-auto flex-wrap md:flex-nowrap'):
                
                # Left Column (Video - dominates screen)
                with ui.column().classes('flex-[4] h-full relative min-w-[400px] flex flex-col gap-3'):
                    with video_feed_card('Outpatient Triage Camera 01'):
                        # Native browser MJPEG handling. No WebSockets, zero UI overhead.
                        ui.element('img').props('src="/camera/stream"').classes('absolute inset-0 w-full h-full object-cover z-10')
                        
                        # Source & Tracker Overlay Badges
                        with ui.row().classes('absolute bottom-4 left-4 z-20 items-center gap-2 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-slate-700'):
                            ui.element('span').classes('w-2 h-2 rounded-full bg-emerald-400 animate-pulse')
                            active_source_label = ui.label('Webcam 0').classes('text-xs font-medium text-slate-200')

                        with ui.row().classes('absolute bottom-4 right-4 z-20 items-center gap-2 bg-indigo-950/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-indigo-700/50'):
                            ui.icon('tune').classes('text-xs text-indigo-400')
                            tracker_badge = ui.label('ByteTrack [Auto]').classes('text-xs font-mono font-medium text-indigo-300')

                        def update_tracker_badge():
                            try:
                                det = get_detector()
                                if det and hasattr(det, 'current_tracker_status_label'):
                                    tracker_badge.text = det.current_tracker_status_label
                            except Exception:
                                pass

                        ui.timer(1.0, update_tracker_badge)


                    # Controls directly under Video Feed
                    with ui.row().classes('w-full items-center justify-between gap-4 p-2 z-30 flex-wrap sm:flex-nowrap'):
                        async def turn_on_webcam():
                            detector = get_detector()
                            if not detector:
                                ui.notify('Detection engine initializing...', type='warning')
                                return
                            
                            btn_webcam.props('loading')
                            success, label_or_err = await run.io_bound(lambda: detector.change_source("0"))
                            btn_webcam.props(remove='loading')

                            if success:
                                active_source_label.text = label_or_err
                                ui.notify('Live Camera (Webcam 0) activated', type='positive', icon='videocam')
                            else:
                                ui.notify(f'Webcam error: {label_or_err}', type='negative', icon='videocam_off')

                        btn_webcam = ui.button('Turn On Live Camera', on_click=turn_on_webcam) \
                            .props('unelevated rounded icon=videocam') \
                            .classes('px-5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs shadow-lg shadow-emerald-950/50')

                        # Model Weights Selector Dropdown
                        async def on_model_change(e):
                            det = get_detector()
                            if not det:
                                return
                            success, msg = await run.io_bound(lambda: det.change_model(e.value))
                            if success:
                                ui.notify(f'AI Model updated: {msg}', type='positive', icon='psychology', timeout=3000)
                            else:
                                ui.notify(f'Model load error: {msg}', type='negative', icon='error')

                        ui.select(
                            options={
                                'models/onnx_versions_fine_tuned/pediatric-model.onnx': '🚀 Pediatric Model (ONNX - 2x Fast)',
                                'models/onnx_versions_fine_tuned/pediatric-kids-only.onnx': '👶 Kids-Only Model (ONNX - 2x Fast)',
                                'models/onnx_versions_fine_tuned/pediatric-smaller-dataset-trained.onnx': '🔬 Smaller Dataset (ONNX - 2x Fast)',
                                'models/fine_tuned/pediatric-model.pt': '🧠 Pediatric Fine-Tuned (PyTorch)',
                                'models/fine_tuned/pediatric-kids-only.pt': '👶 Kids-Only Model (PyTorch)',
                                'models/fine_tuned/pediatric-smaller-dataset-trained.pt': '🔬 Smaller Dataset Model (PyTorch)',
                                'models/base_model/yolo26s.pt': '⚡ Base YOLO26 Small Model'
                            },
                            value='models/fine_tuned/pediatric-model.pt',
                            on_change=on_model_change
                        ).props('dense outlined dark rounded').classes('text-xs bg-slate-900 border-slate-700 min-w-[230px]')

                        # Tracker Algorithm Selector Dropdown
                        async def on_tracker_change(e):
                            det = get_detector()
                            if not det:
                                return
                            success, msg = det.change_tracker_mode(e.value)
                            if success:
                                tracker_badge.text = det.current_tracker_status_label
                                ui.notify(msg, type='positive', icon='tune', timeout=3000)

                        ui.select(
                            options={
                                'auto': '⚡ Auto Switch (Situation Aware)',
                                'bytetrack': '🎯 ByteTrack (Baseline)',
                                'botsort': '📹 BoT-SORT (Motion Comp)',
                                'ocsort': '🔄 OC-SORT (Non-Linear)',
                                'fasttracker': '👥 FastTracker (Occlusion Aware)'
                            },
                            value='auto',
                            on_change=on_tracker_change
                        ).props('dense outlined dark rounded').classes('text-xs bg-slate-900 border-slate-700 min-w-[210px]')

                        ui.button('External Video Testing', on_click=lambda: ui.navigate.to('/video-test')) \
                            .props('outline rounded icon=science') \
                            .classes('px-5 py-2 border-slate-700 text-indigo-400 hover:bg-slate-800 hover:text-indigo-300 font-bold text-xs')



                # Right Column (Telemetry - compact sidebar)
                with ui.column().classes('flex-[1] h-full flex flex-col gap-4 overflow-y-auto min-w-[280px]'):
                    alert_banner('Capacity Warning', 'Pediatric load exceeds standard waiting capacity. Consider dispatching additional triage staff.', state, 'overcrowding_alert')
                    
                    with ui.row().classes('w-full justify-between items-center mb-0 mt-2'):
                        ui.label('Real-Time Telemetry').classes('text-2xl font-bold tracking-tight text-white')
                        # Live blip
                        with ui.element('span').classes('flex h-3 w-3 relative'):
                            ui.element('span').classes('animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75')
                            ui.element('span').classes('relative inline-flex rounded-full h-3 w-3 bg-emerald-600')
                    
                    with ui.row().classes('w-full gap-4 flex-nowrap'):
                        with ui.element('div').classes('flex-1'):
                            stat_card('Pediatric', state, 'current_children', is_primary=True)
                        with ui.element('div').classes('flex-1'):
                            stat_card('Adult', state, 'current_adults', is_primary=False)
                    
                    # Daily Aggregates (flex-grow so it pushes the button to the bottom)
                    with ui.column().classes('w-full flex-grow bg-slate-900/60 border border-slate-800 rounded-2xl p-6 relative'):
                        ui.label('Daily Aggregates').classes('text-xs font-semibold text-slate-400 tracking-[0.15em] uppercase mb-4 pb-4 border-b border-slate-800 w-full')
                        
                        with ui.column().classes('w-full gap-2 mb-6'):
                            with ui.row().classes('w-full justify-between items-center text-sm flex-nowrap'):
                                ui.label('Pediatric Processed').classes('font-medium text-white whitespace-nowrap')
                                ui.label().bind_text_from(state, 'total_daily_children').classes('font-bold tracking-wide text-white whitespace-nowrap ml-2')
                            p_child = ui.linear_progress(value=0, color='primary').classes('h-2 rounded-full')
                            
                        with ui.column().classes('w-full gap-2'):
                            with ui.row().classes('w-full justify-between items-center text-sm flex-nowrap'):
                                ui.label('Adults Processed').classes('font-medium text-slate-400 whitespace-nowrap')
                                ui.label().bind_text_from(state, 'total_daily_adults').classes('font-bold tracking-wide text-slate-300 whitespace-nowrap ml-2')
                            p_adult = ui.linear_progress(value=0, color='grey-6').classes('h-2 rounded-full')
                    
                    # Generate Report Buttons
                    def handle_download(content: bytes, filename: str, media_type: str):
                        from nicegui import app
                        from pathlib import Path
                        if app.native.main_window:
                            # Running in native PyWebView, save directly to Downloads
                            downloads_dir = Path.home() / 'Downloads'
                            downloads_dir.mkdir(parents=True, exist_ok=True)
                            file_path = downloads_dir / filename
                            with open(file_path, 'wb') as f:
                                f.write(content)
                            ui.notify(f'Saved to {file_path}', type='positive', timeout=4000)
                        else:
                            # Web browser mode
                            ui.download.content(content, filename, media_type=media_type)

                    async def generate_csv():
                        btn_csv.props('loading')
                        await asyncio.sleep(0.5)
                        from src.engine import reporter
                        csv_bytes = await run.io_bound(lambda: reporter.generate_csv(state))
                        handle_download(csv_bytes, 'capacity_report.csv', 'text/csv')
                        btn_csv.props(remove='loading')

                    async def generate_pdf():
                        btn_pdf.props('loading')
                        await asyncio.sleep(0.5)
                        from src.engine import reporter
                        pdf_bytes = await run.io_bound(lambda: reporter.generate_pdf(state))
                        handle_download(pdf_bytes, 'capacity_report.pdf', 'application/pdf')
                        btn_pdf.props(remove='loading')

                    with ui.row().classes('w-full gap-2 mt-2'):
                        btn_csv = ui.button('CSV Report', on_click=generate_csv) \
                            .props('outline rounded size=lg icon=table_view') \
                            .classes('flex-1 font-bold border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white transition-colors')
                        btn_pdf = ui.button('PDF Report', on_click=generate_pdf) \
                            .props('outline rounded size=lg icon=picture_as_pdf') \
                            .classes('flex-1 font-bold border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white transition-colors')
