from nicegui import ui, run
import asyncio
from src.ui.components import (
    page_container, 
    navbar,
    video_feed_card,
    stat_card
)
from src.state.telemetry import TelemetryState
from src.engine.detector import Detector
from src.engine.stream_resolver import PRESET_TEST_STREAMS

from typing import Callable, Optional

def register_evaluation(state: TelemetryState, get_detector: Callable[[], Optional[Detector]]):
    @ui.page('/video-test')
    def evaluation_page():
        with page_container():
            with navbar('External Video Evaluation Suite'):
                pass
            
            with ui.row().classes('w-full h-full flex-grow p-4 md:p-6 gap-6 items-stretch overflow-y-auto flex-wrap md:flex-nowrap'):
                
                # Left Column (Video Feed & Source Controls)
                with ui.column().classes('flex-[4] h-full relative min-w-[400px] flex flex-col gap-4'):
                    with video_feed_card('Model Evaluation Feed'):
                        ui.element('img').props('src="/camera/stream"').classes('absolute inset-0 w-full h-full object-cover z-10')
                        
                        # Active Stream Status & Tracker Badges
                        with ui.row().classes('absolute bottom-4 left-4 z-20 items-center gap-2 bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-slate-700'):
                            ui.element('span').classes('w-2 h-2 rounded-full bg-indigo-400 animate-pulse')
                            active_source_label = ui.label('Select Source Below').classes('text-xs font-medium text-slate-200')

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


                    # Video Source Switcher Card
                    with ui.card().classes('w-full p-6 rounded-2xl bg-slate-900/90 border border-slate-800 flex flex-col gap-4 z-30 shadow-xl'):
                        with ui.row().classes('w-full justify-between items-center flex-wrap gap-2'):
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('science').classes('text-indigo-400 text-xl')
                                ui.label('Evaluation Video Source').classes('text-sm font-bold tracking-wider text-slate-200 uppercase')
                            
                            with ui.row().classes('items-center gap-2 flex-wrap'):
                                # Model selector dropdown for evaluation lab
                                async def on_eval_model_change(e):
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
                                    on_change=on_eval_model_change
                                ).props('dense outlined dark rounded').classes('text-xs bg-slate-950 border-slate-700 min-w-[230px]')

                                # Tracker selector dropdown for evaluation lab
                                async def on_eval_tracker_change(e):
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
                                    on_change=on_eval_tracker_change
                                ).props('dense outlined dark rounded').classes('text-xs bg-slate-950 border-slate-700 min-w-[210px]')



                        preset_options = {p['url']: p['name'] for p in PRESET_TEST_STREAMS}
                        preset_options['custom'] = 'Custom Video URL or File Path...'

                        def on_preset_change(e):
                            val = e.value
                            if val != 'custom':
                                custom_input.value = val

                        async def apply_source():
                            target = custom_input.value.strip()
                            detector = get_detector()
                            if not detector:
                                ui.notify('Detection engine initializing...', type='warning')
                                return
                            
                            btn_load.props('loading')
                            success, label_or_err = await run.io_bound(lambda: detector.change_source(target))
                            btn_load.props(remove='loading')

                            if success:
                                active_source_label.text = label_or_err
                                ui.notify(f'Active video source loaded: {label_or_err}', type='positive', icon='check_circle')
                            else:
                                ui.notify(f'Stream error: {label_or_err}', type='negative', icon='error')

                        with ui.row().classes('w-full items-center gap-3 flex-wrap md:flex-nowrap'):
                            preset_select = ui.select(
                                options=preset_options,
                                value=PRESET_TEST_STREAMS[0]['url'],
                                on_change=on_preset_change
                            ).props('dark dense outlined').classes('min-w-[220px] flex-1 text-sm')

                            custom_input = ui.input(
                                placeholder='Paste video file path, YouTube URL, or RTSP link...'
                            ).props('dark dense outlined icon=folder_open').classes('flex-[2] min-w-[280px] text-sm')
                            custom_input.value = PRESET_TEST_STREAMS[0]['url']

                            btn_load = ui.button('Load Stream', on_click=apply_source) \
                                .props('unelevated dense icon=play_arrow') \
                                .classes('px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-xl shadow-lg shadow-indigo-950/50')

                # Right Column (Evaluation Telemetry Stats)
                with ui.column().classes('flex-[1] h-full flex flex-col gap-4 overflow-y-auto min-w-[280px]'):
                    with ui.row().classes('w-full justify-between items-center mb-0 mt-2'):
                        ui.label('Evaluation Telemetry').classes('text-2xl font-bold tracking-tight text-white')
                        ui.icon('analytics').classes('text-indigo-400 text-2xl')

                    with ui.row().classes('w-full gap-4 flex-nowrap'):
                        with ui.element('div').classes('flex-1'):
                            stat_card('Active Pediatric', state, 'current_children', is_primary=True)
                        with ui.element('div').classes('flex-1'):
                            stat_card('Active Adults', state, 'current_adults', is_primary=False)

                    with ui.card().classes('w-full p-6 rounded-2xl bg-slate-900/60 border border-slate-800 gap-3'):
                        ui.label('Quick Controls').classes('text-xs font-semibold text-slate-400 tracking-wider uppercase mb-2')
                        
                        async def reset_webcam():
                            detector = get_detector()
                            if detector:
                                success, label = await run.io_bound(lambda: detector.change_source("0"))
                                if success:
                                    active_source_label.text = label
                                    custom_input.value = "0"
                                    ui.notify('Reset to Webcam 0', type='info')

                        ui.button('Reset to Webcam 0', on_click=reset_webcam) \
                            .props('outline dense icon=videocam') \
                            .classes('w-full text-slate-300 border-slate-700 hover:bg-slate-800')

                        ui.button('Return to Dashboard', on_click=lambda: ui.navigate.to('/dashboard')) \
                            .props('unelevated dense icon=dashboard') \
                            .classes('w-full bg-slate-800 hover:bg-slate-700 text-slate-200 mt-2')
