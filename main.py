import math
import wave
import threading
import time
import json
import os
import copy
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pygame

# INICJALIZACJA AUDIO (zabezpieczona przed brakiem kart dźwiękowych/Wine)
try:
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
except Exception:
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    pygame.mixer.init()

# GRAPHITE THEME
COLOR_BG = "#1e222b"
COLOR_PANEL = "#282d3b"
COLOR_SURFACE = "#323849"
COLOR_GRID_DARK = "#232733"
COLOR_GRID_LIGHT = "#2a2f3d"
COLOR_LINE = "#3d4559"
COLOR_TEXT = "#e1e6f0"
COLOR_ACCENT = "#00d2ff"
COLOR_NOTE = "#ff9f43"
COLOR_NOTE_BORDER = "#ee5253"
COLOR_PATTERN_CLIP = "#2ecc71"
COLOR_PATTERN_BORDER = "#27ae60"
COLOR_PLAYHEAD = "#e84118"
COLOR_KEY_WHITE = "#dcdde1"
COLOR_KEY_BLACK = "#2f3640"
COLOR_ACTIVE_TOOL = "#00a8ff"
COLOR_WAVE_BG = "#12151c"
COLOR_WAVE_LINE = "#a29bfe"

INSTRUMENT_TYPES = ["saw", "square", "sine", "triangle", "noise", "sample", "json_plugin"]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ALL_NOTES = []
for octave in range(5, 1, -1):
    for name in reversed(NOTE_NAMES):
        ALL_NOTES.append((name, octave))

def note_to_freq(note_tuple):
    name, octave = note_tuple
    semitones = NOTE_NAMES.index(name)
    n = semitones + (octave - 4) * 12 - 9
    return 440.0 * (2 ** (n / 12.0))

def resample_audio(audio_data, pitch_factor):
    if pitch_factor == 1.0 or len(audio_data) == 0:
        return audio_data
    old_length = len(audio_data)
    new_length = int(old_length / pitch_factor)
    if new_length <= 0:
        return np.zeros((0, 2), dtype=np.int16)
    
    indices = np.linspace(0, old_length - 1, new_length)
    left = np.interp(indices, np.arange(old_length), audio_data[:, 0])
    right = np.interp(indices, np.arange(old_length), audio_data[:, 1])
    return np.column_stack((left, right)).astype(np.int16)

def generate_osc_wave(wave_type, freq, t, n_samples):
    if wave_type == "sine":
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == "square":
        return np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == "saw":
        return 2 * (t * freq - np.floor(0.5 + t * freq))
    elif wave_type == "triangle":
        return 2 * np.abs(2 * (t * freq - np.floor(0.5 + t * freq))) - 1
    elif wave_type == "noise":
        return np.random.uniform(-0.6, 0.6, n_samples)
    return np.sin(2 * np.pi * freq * t)

def process_json_plugin(plugin_data, base_freq, duration_sec, sample_rate=44100):
    n_samples = int(sample_rate * duration_sec)
    if n_samples <= 0:
        return np.zeros(0)

    t = np.linspace(0, duration_sec, n_samples, False)
    settings = plugin_data.get("settings", {})

    osc1_cfg = settings.get("osc1", {})
    f1 = base_freq * (2 ** osc1_cfg.get("octave", 0)) * (2 ** (osc1_cfg.get("detune", 0) / 12.0))
    sig1 = generate_osc_wave(osc1_cfg.get("wave", "sine"), f1, t, n_samples) * osc1_cfg.get("vol", 1.0)

    osc2_cfg = settings.get("osc2", {})
    if osc2_cfg.get("enable", False):
        f2 = base_freq * (2 ** osc2_cfg.get("octave", 0)) * (2 ** (osc2_cfg.get("detune", 0) / 12.0))
        sig2 = generate_osc_wave(osc2_cfg.get("wave", "sine"), f2, t, n_samples) * osc2_cfg.get("vol", 1.0)
        signal = sig1 + sig2
    else:
        signal = sig1

    env_cfg = settings.get("env", {"a": 0.01, "d": 0.1, "s": 0.8, "r": 0.1})
    a_s = int(env_cfg.get("a", 0.01) * sample_rate)
    d_s = int(env_cfg.get("d", 0.1) * sample_rate)
    s_val = env_cfg.get("s", 0.8)
    r_s = int(env_cfg.get("r", 0.1) * sample_rate)

    env = np.ones(n_samples)
    idx = 0

    if a_s > 0:
        len_a = min(a_s, n_samples)
        env[:len_a] = np.linspace(0.0, 1.0, len_a)
        idx += len_a

    if d_s > 0 and idx < n_samples:
        len_d = min(d_s, n_samples - idx)
        env[idx:idx+len_d] = np.linspace(1.0, s_val, len_d)
        idx += len_d

    if idx < n_samples:
        env[idx:] = s_val

    if r_s > 0 and n_samples > r_s:
        env[-r_s:] *= np.linspace(1.0, 0.0, r_s)

    signal = signal * env

    flt_cfg = settings.get("filter", {})
    cutoff = flt_cfg.get("cutoff", 5000)
    flt_type = flt_cfg.get("type", "lowpass")
    rc = 1.0 / (2 * np.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    filtered = np.zeros_like(signal)
    if flt_type == "highpass":
        alpha = rc / (rc + dt)
        last_in = signal[0] if len(signal) > 0 else 0
        last_out = 0
        for i in range(len(signal)):
            out = alpha * (last_out + signal[i] - last_in)
            filtered[i] = out
            last_in = signal[i]
            last_out = out
    else:
        last_out = 0
        for i in range(len(signal)):
            last_out = last_out + alpha * (signal[i] - last_out)
            filtered[i] = last_out
    signal = filtered

    fx_cfg = settings.get("fx", {})
    dist = fx_cfg.get("dist", 0)
    if dist > 0:
        drive = 1.0 + dist * 10.0
        signal = np.tanh(signal * drive)

    delay = fx_cfg.get("delay", 0)
    if delay > 0:
        delay_samples = int(0.2 * sample_rate)
        delayed_signal = np.zeros_like(signal)
        if len(signal) > delay_samples:
            delayed_signal[delay_samples:] = signal[:-delay_samples] * delay
        signal += delayed_signal

    return signal

def generate_wave_signal(wave_type, freq, duration_sec, sample_rate=44100, volume=0.3, sample_data=None, next_freq=None, is_slide=False, json_plugin=None):
    n_samples = int(sample_rate * duration_sec)
    if n_samples <= 0:
        return np.zeros((0, 2), dtype=np.int16)

    if wave_type == "json_plugin" and json_plugin is not None:
        raw_sig = process_json_plugin(json_plugin, freq, duration_sec, sample_rate)
        raw_sig = np.clip(raw_sig, -1.0, 1.0) * volume
        int_sig = (raw_sig * 32767).astype(np.int16)
        return np.column_stack((int_sig, int_sig))

    t = np.linspace(0, duration_sec, n_samples, False)

    if is_slide and next_freq is not None and wave_type != "sample":
        freq_array = np.linspace(freq, next_freq, n_samples)
        phase = 2 * np.pi * np.cumsum(freq_array) / sample_rate
        if wave_type == "sine":
            signal = np.sin(phase)
        elif wave_type == "square":
            signal = np.sign(np.sin(phase))
        elif wave_type == "saw":
            signal = 2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))
        elif wave_type == "triangle":
            signal = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))) - 1
        else:
            signal = np.sin(phase)
    elif wave_type == "sample" and sample_data is not None:
        base_freq = 261.63
        pitch_factor = freq / base_freq
        resampled = resample_audio(sample_data, pitch_factor)
        if len(resampled) > n_samples:
            int_signal_stereo = resampled[:n_samples]
        else:
            padding = np.zeros((n_samples - len(resampled), 2), dtype=np.int16)
            int_signal_stereo = np.vstack((resampled, padding))
        return (int_signal_stereo * volume).astype(np.int16)
    else:
        signal = generate_osc_wave(wave_type, freq, t, n_samples)

    env = np.ones(n_samples)
    attack = min(int(0.005 * sample_rate), n_samples // 4)
    release = min(int(0.02 * sample_rate), n_samples // 4)

    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack)
    if release > 0:
        env[-release:] = np.linspace(1.0, 0.0, release)

    signal = signal * env * volume
    int_signal = (signal * 32767).astype(np.int16)
    return np.column_stack((int_signal, int_signal))

class NoteBlock:
    def __init__(self, pitch_idx, start_step, duration_steps=2):
        self.pitch_idx = pitch_idx
        self.start_step = start_step
        self.duration = duration_steps

    def to_dict(self):
        return {"pitch_idx": self.pitch_idx, "start_step": self.start_step, "duration": self.duration}

    @staticmethod
    def from_dict(data):
        return NoteBlock(data["pitch_idx"], data["start_step"], data["duration"])

class PatternClip:
    def __init__(self, pattern_idx, start_step, duration_steps=16):
        self.pattern_idx = pattern_idx
        self.start_step = start_step
        self.duration = duration_steps

    def to_dict(self):
        return {"pattern_idx": self.pattern_idx, "start_step": self.start_step, "duration": self.duration}

    @staticmethod
    def from_dict(data):
        return PatternClip(data["pattern_idx"], data["start_step"], data["duration"])

class Track:
    def __init__(self, name="Synth Lead", wave_type="saw"):
        self.name = name
        self.wave_type = wave_type
        self.volume = 0.3
        self.sample_path = None
        self.sample_data = None
        self.json_plugin_data = None

    def load_sample(self, file_path):
        try:
            snd = pygame.mixer.Sound(file_path)
            arr = pygame.sndarray.array(snd)
            if len(arr.shape) == 1:
                arr = np.column_stack((arr, arr))
            self.sample_data = arr
            self.sample_path = file_path
            self.wave_type = "sample"
            self.name = os.path.splitext(os.path.basename(file_path))[0]
            return True
        except Exception as e:
            messagebox.showerror("Error Loading Sample", str(e))
            return False

    def load_json_plugin(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.json_plugin_data = json.load(f)
            self.wave_type = "json_plugin"
            default_name = os.path.splitext(os.path.basename(file_path))[0]
            self.name = self.json_plugin_data.get("name", default_name)
            return True
        except Exception as e:
            messagebox.showerror("Error Loading Plugin", str(e))
            return False

    def to_dict(self):
        return {
            "name": self.name,
            "wave_type": self.wave_type,
            "volume": self.volume,
            "sample_path": self.sample_path,
            "json_plugin_data": self.json_plugin_data
        }

    @staticmethod
    def from_dict(data):
        t = Track(data["name"], data.get("wave_type", "saw"))
        t.volume = data.get("volume", 0.3)
        sp = data.get("sample_path")
        if sp and os.path.exists(sp):
            t.load_sample(sp)
        t.json_plugin_data = data.get("json_plugin_data")
        return t

class PatternData:
    def __init__(self, name="Pattern 1", total_steps=16):
        self.name = name
        self.total_steps = total_steps
        self.track_notes = {}
        self.track_instruments = {}
        self.track_slides = {}

    def to_dict(self):
        tn_dict = {str(k): [n.to_dict() for n in v] for k, v in self.track_notes.items()}
        ti_dict = {str(k): v for k, v in self.track_instruments.items()}
        ts_dict = {str(k): v for k, v in self.track_slides.items()}
        return {"name": self.name, "total_steps": self.total_steps, "track_notes": tn_dict, "track_instruments": ti_dict, "track_slides": ts_dict}

    @staticmethod
    def from_dict(data):
        p = PatternData(data["name"], data["total_steps"])
        for k, v in data.get("track_notes", {}).items():
            p.track_notes[int(k)] = [NoteBlock.from_dict(n) for n in v]
        for k, v in data.get("track_instruments", {}).items():
            p.track_instruments[int(k)] = v
        for k, v in data.get("track_slides", {}).items():
            p.track_slides[int(k)] = v
        return p

class FLStudioDAW(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LunaBox Beta")
        self.geometry("1400x850")
        self.configure(bg=COLOR_BG)

        self.bpm = 120
        self.playlist_steps = 128
        self.current_step = 0
        self.is_playing = False
        self.play_mode = "PAT"

        self.active_track_idx = 0
        self.active_pattern_idx = -1
        self.active_tool = "paint"

        self.key_width = 80
        self.row_height = 20
        self.step_width = 24
        self.playlist_row_h = 40

        self.tracks = [
            Track("Lead Synth", "saw"),
            Track("Bassline", "square"),
            Track("Pluck Sound", "triangle"),
            Track("Noise Perc", "noise")
        ]

        self.patterns = []
        self.playlist_tracks = [[] for _ in range(32)]

        self.selected_note = None
        self.clipboard_notes = []
        self.undo_stack = []
        self.selected_clip = None
        self.is_resizing = False
        self.is_moving_note = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.note_start_step_orig = 0
        self.note_start_pitch_orig = 0

        self.init_menu()
        self.init_ui()
        self.bind_shortcuts()
        self.add_new_pattern()

        self.last_played_step = -1
        self.update_playhead_loop()

    def bind_shortcuts(self):
        self.bind("<Control-s>", lambda event: self.save_project())
        self.bind("<Control-S>", lambda event: self.save_project())
        self.bind("<Control-c>", lambda event: self.copy_notes())
        self.bind("<Control-C>", lambda event: self.copy_notes())
        self.bind("<Control-v>", lambda event: self.paste_notes())
        self.bind("<Control-V>", lambda event: self.paste_notes())
        self.bind("<Control-y>", lambda event: self.paste_notes())
        self.bind("<Control-Y>", lambda event: self.paste_notes())
        self.bind("<Control-z>", lambda event: self.undo())
        self.bind("<Control-Z>", lambda event: self.undo())

    def save_undo_state(self):
        if self.active_pattern_idx >= 0 and self.patterns:
            state = copy.deepcopy(self.patterns[self.active_pattern_idx].to_dict())
            self.undo_stack.append((self.active_pattern_idx, state))
            if len(self.undo_stack) > 30:
                self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            return
        pat_idx, state = self.undo_stack.pop()
        if 0 <= pat_idx < len(self.patterns):
            self.patterns[pat_idx] = PatternData.from_dict(state)
            self.draw_piano_roll()
            self.update_waveform_display()

    def copy_notes(self):
        if self.active_pattern_idx < 0 or not self.patterns:
            return
        active_pat = self.patterns[self.active_pattern_idx]
        notes = active_pat.track_notes.get(self.active_track_idx, [])
        if self.selected_note:
            self.clipboard_notes = [self.selected_note.to_dict()]
        else:
            self.clipboard_notes = [n.to_dict() for n in notes]

    def paste_notes(self):
        if not self.clipboard_notes or self.active_pattern_idx < 0 or not self.patterns:
            return

        self.save_undo_state()
        active_pat = self.patterns[self.active_pattern_idx]
        notes = active_pat.track_notes.setdefault(self.active_track_idx, [])

        min_start = min(n["start_step"] for n in self.clipboard_notes)
        offset_step = self.current_step

        for n_dict in self.clipboard_notes:
            new_step = offset_step + (n_dict["start_step"] - min_start)
            new_note = NoteBlock(n_dict["pitch_idx"], new_step, n_dict["duration"])
            notes.append(new_note)

        self.check_and_expand_pattern()
        self.draw_piano_roll()

    def init_menu(self):
        self.menubar = tk.Menu(self, bg=COLOR_PANEL, fg=COLOR_TEXT)
        self.project_menu = tk.Menu(self.menubar, tearoff=0, bg=COLOR_PANEL, fg=COLOR_TEXT)
        self.project_menu.add_command(label="Save Project (.lbp) (Ctrl+S)", command=self.save_project)
        self.project_menu.add_command(label="Load Project (.lbp)", command=self.load_project)
        self.project_menu.add_separator()
        self.project_menu.add_command(label="Export Audio (.wav)", command=self.export_audio)
        self.menubar.add_cascade(label="Project", menu=self.project_menu)
        self.config(menu=self.menubar)

    def init_ui(self):
        self.top_bar = tk.Frame(self, bg=COLOR_PANEL, height=50)
        self.top_bar.pack(side=tk.TOP, fill=tk.X)

        self.btn_play = tk.Button(self.top_bar, text="▶ PLAY", bg=COLOR_SURFACE, fg="#2ecc71", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=10, pady=8)

        self.btn_stop = tk.Button(self.top_bar, text="■ STOP", bg=COLOR_SURFACE, fg="#e74c3c", font=("Segoe UI", 10, "bold"), relief=tk.FLAT, command=self.stop_play)
        self.btn_stop.pack(side=tk.LEFT, padx=5, pady=8)

        self.btn_mode = tk.Button(self.top_bar, text="PAT", bg=COLOR_ACCENT, fg=COLOR_BG, font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=self.toggle_mode)
        self.btn_mode.pack(side=tk.LEFT, padx=10, pady=8)

        tk.Label(self.top_bar, text="BPM:", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(10, 2))
        self.bpm_spin = tk.Spinbox(self.top_bar, from_=40, to=240, width=5, bg=COLOR_SURFACE, fg=COLOR_ACCENT, relief=tk.FLAT)
        self.bpm_spin.delete(0, "end")
        self.bpm_spin.insert(0, str(self.bpm))
        self.bpm_spin.pack(side=tk.LEFT, pady=8)

        tk.Label(self.top_bar, text="PATTERN:", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(15, 2))
        self.pat_cb = ttk.Combobox(self.top_bar, values=[], state="readonly", width=12)
        self.pat_cb.pack(side=tk.LEFT, pady=8)
        self.pat_cb.bind("<<ComboboxSelected>>", self.on_pattern_change)

        self.btn_add_pat = tk.Button(self.top_bar, text="+ Pattern", bg=COLOR_SURFACE, fg=COLOR_ACCENT, font=("Segoe UI", 8, "bold"), relief=tk.FLAT, command=self.add_new_pattern)
        self.btn_add_pat.pack(side=tk.LEFT, padx=3, pady=8)

        self.wave_canvas = tk.Canvas(self.top_bar, width=160, height=32, bg=COLOR_WAVE_BG, highlightthickness=1, highlightbackground=COLOR_LINE)
        self.wave_canvas.pack(side=tk.LEFT, padx=15, pady=8)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_piano = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(self.tab_piano, text="  PIANO ROLL  ")

        self.pr_toolbar = tk.Frame(self.tab_piano, bg=COLOR_PANEL, height=32)
        self.pr_toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_tool_paint = tk.Button(self.pr_toolbar, text="Paint", bg=COLOR_ACTIVE_TOOL, fg="#ffffff", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, command=lambda: self.set_tool("paint"))
        self.btn_tool_paint.pack(side=tk.LEFT, padx=2, pady=3)

        self.btn_tool_delete = tk.Button(self.pr_toolbar, text="Delete", bg=COLOR_SURFACE, fg=COLOR_TEXT, font=("Segoe UI", 8, "bold"), relief=tk.FLAT, command=lambda: self.set_tool("delete"))
        self.btn_tool_delete.pack(side=tk.LEFT, padx=2, pady=3)

        self.btn_tool_select = tk.Button(self.pr_toolbar, text="Select", bg=COLOR_SURFACE, fg=COLOR_TEXT, font=("Segoe UI", 8, "bold"), relief=tk.FLAT, command=lambda: self.set_tool("select"))
        self.btn_tool_select.pack(side=tk.LEFT, padx=2, pady=3)

        tk.Label(self.pr_toolbar, text="TRACK:", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(10, 2))
        self.track_cb = ttk.Combobox(self.pr_toolbar, values=[t.name for t in self.tracks], state="readonly", width=14)
        self.track_cb.current(0)
        self.track_cb.pack(side=tk.LEFT, pady=3)
        self.track_cb.bind("<<ComboboxSelected>>", self.on_track_change)

        tk.Label(self.pr_toolbar, text="WAVE/PLUGIN:", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(8, 2))
        self.wave_cb = ttk.Combobox(self.pr_toolbar, values=INSTRUMENT_TYPES, state="readonly", width=11)
        self.wave_cb.set(self.tracks[self.active_track_idx].wave_type)
        self.wave_cb.pack(side=tk.LEFT, pady=3)
        self.wave_cb.bind("<<ComboboxSelected>>", self.on_wave_change)

        self.btn_load_plugin = tk.Button(self.pr_toolbar, text="[Plug] Load Plugin (.plug)", bg=COLOR_SURFACE, fg="#f1c40f", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, command=self.load_plugin_file)
        self.btn_load_plugin.pack(side=tk.LEFT, padx=5, pady=3)

        self.btn_load_sample = tk.Button(self.pr_toolbar, text="[Folder] Load Sample", bg=COLOR_SURFACE, fg=COLOR_ACCENT, font=("Segoe UI", 8, "bold"), relief=tk.FLAT, command=self.load_sample_file)
        self.btn_load_sample.pack(side=tk.LEFT, padx=2, pady=3)

        self.slide_var = tk.BooleanVar(value=False)
        self.chk_slide = tk.Checkbutton(self.pr_toolbar, text="Slide", variable=self.slide_var, bg=COLOR_PANEL, fg=COLOR_TEXT, selectcolor=COLOR_SURFACE, font=("Segoe UI", 8, "bold"), command=self.on_slide_toggle)
        self.chk_slide.pack(side=tk.LEFT, padx=10, pady=3)

        self.pr_main_frame = tk.Frame(self.tab_piano, bg=COLOR_BG)
        self.pr_main_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.keys_canvas = tk.Canvas(self.pr_main_frame, width=self.key_width, bg=COLOR_PANEL, highlightthickness=0)
        self.keys_canvas.pack(side=tk.LEFT, fill=tk.Y)

        self.pr_frame = tk.Frame(self.pr_main_frame, bg=COLOR_BG)
        self.pr_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.pr_canvas = tk.Canvas(self.pr_frame, bg=COLOR_BG, highlightthickness=0)
        self.pr_v_scroll = tk.Scrollbar(self.pr_frame, orient=tk.VERTICAL, command=self.sync_pr_scroll)
        self.pr_h_scroll = tk.Scrollbar(self.pr_frame, orient=tk.HORIZONTAL, command=self.pr_canvas.xview)

        self.pr_canvas.configure(yscrollcommand=self.pr_v_scroll.set, xscrollcommand=self.pr_h_scroll.set)
        self.pr_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.pr_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.pr_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.pr_canvas.bind("<ButtonPress-1>", self.on_piano_roll_click)
        self.pr_canvas.bind("<B1-Motion>", self.on_piano_roll_drag)
        self.pr_canvas.bind("<ButtonRelease-1>", self.on_piano_roll_release)
        self.pr_canvas.bind("<ButtonPress-3>", self.on_piano_roll_right_click)

        self.tab_playlist = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(self.tab_playlist, text="  PLAYLIST  ")

        self.pl_headers = tk.Canvas(self.tab_playlist, width=100, bg=COLOR_PANEL, highlightthickness=0)
        self.pl_headers.pack(side=tk.LEFT, fill=tk.Y)

        self.pl_frame = tk.Frame(self.tab_playlist, bg=COLOR_BG)
        self.pl_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.pl_canvas = tk.Canvas(self.pl_frame, bg=COLOR_BG, highlightthickness=0)
        self.pl_v_scroll = tk.Scrollbar(self.pl_frame, orient=tk.VERTICAL, command=self.sync_pl_scroll)
        self.pl_h_scroll = tk.Scrollbar(self.pl_frame, orient=tk.HORIZONTAL, command=self.pl_canvas.xview)

        self.pl_canvas.configure(yscrollcommand=self.pl_v_scroll.set, xscrollcommand=self.pl_h_scroll.set)
        self.pl_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.pl_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.pl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.pl_canvas.bind("<ButtonPress-1>", self.on_playlist_click)
        self.pl_canvas.bind("<B1-Motion>", self.on_playlist_drag)
        self.pl_canvas.bind("<ButtonRelease-1>", self.on_playlist_release)
        self.pl_canvas.bind("<ButtonPress-3>", self.on_playlist_right_click)

        self.draw_piano_keys()
        self.draw_piano_roll()
        self.draw_playlist_headers()
        self.draw_playlist()
        self.update_waveform_display()

    def load_plugin_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Plugin File", "*.plug"), ("JSON Plugin", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            track = self.tracks[self.active_track_idx]
            if track.load_json_plugin(file_path):
                self.wave_cb.set("json_plugin")
                self.on_wave_change(None)
                self.track_cb["values"] = [t.name for t in self.tracks]
                self.track_cb.current(self.active_track_idx)

    def load_sample_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.ogg"), ("All Files", "*.*")])
        if file_path:
            track = self.tracks[self.active_track_idx]
            if track.load_sample(file_path):
                self.wave_cb.set("sample")
                self.on_wave_change(None)
                self.track_cb["values"] = [t.name for t in self.tracks]
                self.track_cb.current(self.active_track_idx)

    def on_slide_toggle(self):
        if self.active_pattern_idx >= 0 and self.patterns:
            pat = self.patterns[self.active_pattern_idx]
            pat.track_slides[self.active_track_idx] = self.slide_var.get()

    def update_waveform_display(self):
        self.wave_canvas.delete("all")
        w, h = 160, 32
        mid = h / 2
        self.wave_canvas.create_line(0, mid, w, mid, fill="#2c3e50", width=1)

        if self.active_pattern_idx < 0 or not self.patterns:
            return

        pat = self.patterns[self.active_pattern_idx]
        wave_type = pat.track_instruments.get(self.active_track_idx, self.tracks[self.active_track_idx].wave_type)

        points = []
        for x in range(w):
            t = (x / w) * 4 * np.pi
            if wave_type == "sine": val = np.sin(t * 3)
            elif wave_type == "square": val = np.sign(np.sin(t * 3))
            elif wave_type == "saw": val = 2 * (t / (2*np.pi) - np.floor(0.5 + t / (2*np.pi)))
            elif wave_type == "triangle": val = 2 * np.abs(2 * (t / (2*np.pi) - np.floor(0.5 + t / (2*np.pi)))) - 1
            elif wave_type == "noise": val = np.random.uniform(-0.8, 0.8)
            elif wave_type == "json_plugin": val = (np.sin(t * 3) + np.sin(t * 6 * 1.05) * 0.5) * 0.6
            else: val = np.sin(t * 3)
            y = mid - (val * (h / 2.5))
            points.append((x, y))

        for i in range(len(points) - 1):
            self.wave_canvas.create_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1], fill=COLOR_WAVE_LINE, width=1.5)

    def play_step_sounds(self, step):
        step_dur = (60.0 / self.bpm) / 4.0

        if self.play_mode == "PAT" and self.active_pattern_idx >= 0 and self.patterns:
            pat = self.patterns[self.active_pattern_idx]
            for tr_idx, notes in pat.track_notes.items():
                if tr_idx < len(self.tracks):
                    wave_type = pat.track_instruments.get(tr_idx, self.tracks[tr_idx].wave_type)
                    is_slide = pat.track_slides.get(tr_idx, False)
                    for note in notes:
                        if note.start_step == step:
                            freq = note_to_freq(ALL_NOTES[note.pitch_idx])
                            dur_sec = note.duration * step_dur
                            next_freq = None
                            if is_slide:
                                sorted_notes = sorted(notes, key=lambda n: n.start_step)
                                for n_next in sorted_notes:
                                    if n_next.start_step > note.start_step:
                                        next_freq = note_to_freq(ALL_NOTES[n_next.pitch_idx])
                                        break
                            audio = generate_wave_signal(wave_type, freq, dur_sec, volume=self.tracks[tr_idx].volume, sample_data=self.tracks[tr_idx].sample_data, next_freq=next_freq, is_slide=is_slide, json_plugin=self.tracks[tr_idx].json_plugin_data)
                            if len(audio) > 0:
                                snd = pygame.sndarray.make_sound(audio)
                                snd.play()

        elif self.play_mode == "SONG":
            for track_clips in self.playlist_tracks:
                for clip in track_clips:
                    if clip.start_step <= step < (clip.start_step + clip.duration):
                        local_step = step - clip.start_step
                        if clip.pattern_idx < len(self.patterns):
                            pat = self.patterns[clip.pattern_idx]
                            for tr_idx, notes in pat.track_notes.items():
                                if tr_idx < len(self.tracks):
                                    wave_type = pat.track_instruments.get(tr_idx, self.tracks[tr_idx].wave_type)
                                    is_slide = pat.track_slides.get(tr_idx, False)
                                    for note in notes:
                                        if note.start_step == local_step:
                                            freq = note_to_freq(ALL_NOTES[note.pitch_idx])
                                            dur_sec = note.duration * step_dur
                                            next_freq = None
                                            if is_slide:
                                                sorted_notes = sorted(notes, key=lambda n: n.start_step)
                                                for n_next in sorted_notes:
                                                    if n_next.start_step > note.start_step:
                                                        next_freq = note_to_freq(ALL_NOTES[n_next.pitch_idx])
                                                        break
                                            audio = generate_wave_signal(wave_type, freq, dur_sec, volume=self.tracks[tr_idx].volume, sample_data=self.tracks[tr_idx].sample_data, next_freq=next_freq, is_slide=is_slide, json_plugin=self.tracks[tr_idx].json_plugin_data)
                                            if len(audio) > 0:
                                                snd = pygame.sndarray.make_sound(audio)
                                                snd.play()

    def update_playhead_loop(self):
        if self.is_playing:
            try: self.bpm = int(self.bpm_spin.get())
            except ValueError: pass

            max_steps = self.playlist_steps
            if self.play_mode == "PAT" and self.active_pattern_idx >= 0 and self.patterns:
                max_steps = self.patterns[self.active_pattern_idx].total_steps

            if self.current_step != self.last_played_step:
                self.play_step_sounds(self.current_step)
                self.last_played_step = self.current_step

            if self.play_mode == "PAT": self.draw_piano_roll()
            else: self.draw_playlist()

            self.current_step = (self.current_step + 1) % max_steps

        step_ms = int((60000 / self.bpm) / 4)
        self.after(step_ms, self.update_playhead_loop)

    def save_project(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".lbp", filetypes=[("LunaDAW Project", "*.lbp")])
        if not file_path: return
        try: self.bpm = int(self.bpm_spin.get())
        except ValueError: pass

        project_data = {
            "bpm": self.bpm,
            "playlist_steps": self.playlist_steps,
            "tracks": [t.to_dict() for t in self.tracks],
            "patterns": [p.to_dict() for p in self.patterns],
            "playlist_tracks": [[c.to_dict() for c in clips] for clips in self.playlist_tracks]
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=4)
        messagebox.showinfo("Project Saved", f"Saved to:\n{file_path}")

    def load_project(self):
        file_path = filedialog.askopenfilename(filetypes=[("LunaDAW Project", "*.lbp")])
        if not file_path: return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                project_data = json.load(f)

            self.bpm = project_data.get("bpm", 120)
            self.bpm_spin.delete(0, "end")
            self.bpm_spin.insert(0, str(self.bpm))

            self.playlist_steps = project_data.get("playlist_steps", 128)
            self.tracks = [Track.from_dict(t) for t in project_data.get("tracks", [])]
            self.patterns = [PatternData.from_dict(p) for p in project_data.get("patterns", [])]

            pl_data = project_data.get("playlist_tracks", [])
            self.playlist_tracks = [[PatternClip.from_dict(c) for c in clips] for clips in pl_data]

            self.track_cb["values"] = [t.name for t in self.tracks]
            if self.tracks:
                self.active_track_idx = 0
                self.track_cb.current(0)

            if self.patterns:
                self.active_pattern_idx = 0
                wt = self.patterns[0].track_instruments.get(0, self.tracks[0].wave_type)
                self.wave_cb.set(wt)
                self.slide_var.set(self.patterns[0].track_slides.get(0, False))
            else:
                self.active_pattern_idx = -1

            self.undo_stack.clear()
            self.update_pattern_combo()
            self.draw_piano_roll()
            self.draw_playlist()
            self.update_waveform_display()
            messagebox.showinfo("Loaded", f"Loaded:\n{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load:\n{str(e)}")

    def export_audio(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Audio", "*.wav")])
        if not file_path: return

        sample_rate = 44100
        step_dur = (60.0 / self.bpm) / 4.0
        total_duration = self.playlist_steps * step_dur
        total_samples = int(total_duration * sample_rate)

        master_buffer = np.zeros((total_samples, 2), dtype=np.float32)

        for track_clips in self.playlist_tracks:
            for clip in track_clips:
                if clip.pattern_idx >= len(self.patterns): continue
                pat = self.patterns[clip.pattern_idx]
                for tr_idx, notes in pat.track_notes.items():
                    if tr_idx >= len(self.tracks): continue
                    wave_type = pat.track_instruments.get(tr_idx, self.tracks[tr_idx].wave_type)
                    is_slide = pat.track_slides.get(tr_idx, False)

                    for note in notes:
                        start_step = clip.start_step + note.start_step
                        start_sec = start_step * step_dur
                        start_sample = int(start_sec * sample_rate)

                        freq = note_to_freq(ALL_NOTES[note.pitch_idx])
                        dur_sec = note.duration * step_dur
                        next_freq = None
                        if is_slide:
                            sorted_notes = sorted(notes, key=lambda n: n.start_step)
                            for n_next in sorted_notes:
                                if n_next.start_step > note.start_step:
                                    next_freq = note_to_freq(ALL_NOTES[n_next.pitch_idx])
                                    break

                        audio_data = generate_wave_signal(wave_type, freq, dur_sec, sample_rate, self.tracks[tr_idx].volume, sample_data=self.tracks[tr_idx].sample_data, next_freq=next_freq, is_slide=is_slide, json_plugin=self.tracks[tr_idx].json_plugin_data)
                        audio_float = audio_data.astype(np.float32) / 32767.0
                        end_sample = min(total_samples, start_sample + len(audio_float))
                        actual_len = end_sample - start_sample

                        if actual_len > 0:
                            master_buffer[start_sample:end_sample] += audio_float[:actual_len]

        master_buffer = np.clip(master_buffer, -1.0, 1.0)
        final_int = (master_buffer * 32767).astype(np.int16)

        with wave.open(file_path, "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(final_int.tobytes())

        messagebox.showinfo("Exported", f"Exported:\n{file_path}")

    def set_tool(self, tool_name):
        self.active_tool = tool_name
        self.btn_tool_paint.config(bg=COLOR_ACTIVE_TOOL if tool_name == "paint" else COLOR_SURFACE)
        self.btn_tool_delete.config(bg=COLOR_ACTIVE_TOOL if tool_name == "delete" else COLOR_SURFACE)
        self.btn_tool_select.config(bg=COLOR_ACTIVE_TOOL if tool_name == "select" else COLOR_SURFACE)

    def add_new_pattern(self):
        pat_num = len(self.patterns) + 1
        new_pat = PatternData(f"Pattern {pat_num}", 16)
        for i, tr in enumerate(self.tracks):
            new_pat.track_notes[i] = []
            new_pat.track_instruments[i] = tr.wave_type
            new_pat.track_slides[i] = False

        self.patterns.append(new_pat)
        self.update_pattern_combo()
        self.active_pattern_idx = len(self.patterns) - 1
        self.pat_cb.current(self.active_pattern_idx)
        self.draw_piano_roll()

    def update_pattern_combo(self):
        names = [p.name for p in self.patterns]
        self.pat_cb["values"] = names
        if self.active_pattern_idx >= 0 and self.active_pattern_idx < len(self.patterns):
            self.pat_cb.current(self.active_pattern_idx)
        else: self.pat_cb.set("")

    def sync_pr_scroll(self, *args):
        self.pr_canvas.yview(*args)
        self.keys_canvas.yview(*args)

    def sync_pl_scroll(self, *args):
        self.pl_canvas.yview(*args)
        self.pl_headers.yview(*args)

    def toggle_mode(self):
        self.play_mode = "SONG" if self.play_mode == "PAT" else "PAT"
        self.btn_mode.config(text=self.play_mode, bg="#f1c40f" if self.play_mode == "SONG" else COLOR_ACCENT)

    def on_pattern_change(self, event):
        if self.pat_cb.current() >= 0:
            self.active_pattern_idx = self.pat_cb.current()
            pat = self.patterns[self.active_pattern_idx]
            wt = pat.track_instruments.get(self.active_track_idx, self.tracks[self.active_track_idx].wave_type)
            self.wave_cb.set(wt)
            self.slide_var.set(pat.track_slides.get(self.active_track_idx, False))
            self.draw_piano_roll()
            self.update_waveform_display()

    def on_track_change(self, event):
        self.active_track_idx = self.track_cb.current()
        if self.active_pattern_idx >= 0 and self.patterns:
            pat = self.patterns[self.active_pattern_idx]
            wt = pat.track_instruments.get(self.active_track_idx, self.tracks[self.active_track_idx].wave_type)
            self.wave_cb.set(wt)
            self.slide_var.set(pat.track_slides.get(self.active_track_idx, False))
        self.draw_piano_roll()
        self.update_waveform_display()

    def on_wave_change(self, event):
        wave_t = self.wave_cb.get()
        if self.active_pattern_idx >= 0 and self.patterns:
            pat = self.patterns[self.active_pattern_idx]
            pat.track_instruments[self.active_track_idx] = wave_t
        self.tracks[self.active_track_idx].wave_type = wave_t
        self.update_waveform_display()

    def draw_piano_keys(self):
        self.keys_canvas.delete("all")
        for i, (name, oct_num) in enumerate(ALL_NOTES):
            y1 = i * self.row_height
            y2 = y1 + self.row_height
            is_black = "#" in name
            bg_col = COLOR_KEY_BLACK if is_black else COLOR_KEY_WHITE
            fg_col = COLOR_TEXT if is_black else "#1e222b"

            self.keys_canvas.create_rectangle(0, y1, self.key_width, y2, fill=bg_col, outline=COLOR_LINE)
            self.keys_canvas.create_text(10, y1 + 10, text=f"{name}{oct_num}", fill=fg_col, anchor="w", font=("Segoe UI", 8, "bold"))

        self.keys_canvas.config(scrollregion=(0, 0, self.key_width, len(ALL_NOTES) * self.row_height))

    def check_and_expand_pattern(self):
        if self.active_pattern_idx < 0: return False
        active_pat = self.patterns[self.active_pattern_idx]
        max_step = active_pat.total_steps

        for tr_notes in active_pat.track_notes.values():
            for note in tr_notes:
                end_step = note.start_step + note.duration
                if end_step >= max_step - 2:
                    max_step = end_step + 16

        if max_step != active_pat.total_steps:
            active_pat.total_steps = max_step
            return True
        return False

    def draw_piano_roll(self):
        self.pr_canvas.delete("all")
        if self.active_pattern_idx < 0 or not self.patterns: return

        active_pat = self.patterns[self.active_pattern_idx]
        total_steps = active_pat.total_steps
        total_w = total_steps * self.step_width
        total_h = len(ALL_NOTES) * self.row_height

        for i, (name, _) in enumerate(ALL_NOTES):
            y1 = i * self.row_height
            y2 = y1 + self.row_height
            bg_col = COLOR_GRID_DARK if "#" in name else COLOR_GRID_LIGHT
            self.pr_canvas.create_rectangle(0, y1, total_w, y2, fill=bg_col, outline="")

        for step in range(total_steps):
            x = step * self.step_width
            is_bar = (step % 4 == 0)
            line_col = COLOR_ACCENT if is_bar else COLOR_LINE
            width = 2 if is_bar else 1
            self.pr_canvas.create_line(x, 0, x, total_h, fill=line_col, width=width)

        notes = active_pat.track_notes.get(self.active_track_idx, [])
        for note in notes:
            x1 = note.start_step * self.step_width
            x2 = x1 + (note.duration * self.step_width)
            y1 = note.pitch_idx * self.row_height
            y2 = y1 + self.row_height

            self.pr_canvas.create_rectangle(x1 + 1, y1 + 1, x2 - 1, y2 - 1, fill=COLOR_NOTE, outline=COLOR_NOTE_BORDER, width=1)
            self.pr_canvas.create_rectangle(x2 - 5, y1 + 1, x2 - 1, y2 - 1, fill=COLOR_ACCENT, outline="")

        if self.play_mode == "PAT":
            px = self.current_step * self.step_width
            self.pr_canvas.create_line(px, 0, px, total_h, fill=COLOR_PLAYHEAD, width=3)

        self.pr_canvas.config(scrollregion=(0, 0, total_w, total_h))
        self.draw_playlist()

    def preview_note_sound(self, pitch_idx):
        freq = note_to_freq(ALL_NOTES[pitch_idx])
        wave_type = "saw"
        if self.active_pattern_idx >= 0 and self.patterns:
            pat = self.patterns[self.active_pattern_idx]
            wave_type = pat.track_instruments.get(self.active_track_idx, self.tracks[self.active_track_idx].wave_type)

        audio_data = generate_wave_signal(wave_type, freq, 0.2, volume=self.tracks[self.active_track_idx].volume, sample_data=self.tracks[self.active_track_idx].sample_data, json_plugin=self.tracks[self.active_track_idx].json_plugin_data)
        sound = pygame.sndarray.make_sound(audio_data)
        sound.play()

    def on_piano_roll_click(self, event):
        if self.active_pattern_idx < 0: self.add_new_pattern()

        x = self.pr_canvas.canvasx(event.x)
        y = self.pr_canvas.canvasy(event.y)

        step = int(x // self.step_width)
        pitch_idx = int(y // self.row_height)
        active_pat = self.patterns[self.active_pattern_idx]

        if not (0 <= pitch_idx < len(ALL_NOTES)): return

        notes = active_pat.track_notes.setdefault(self.active_track_idx, [])
        clicked_note = next((n for n in notes if n.pitch_idx == pitch_idx and n.start_step <= step < (n.start_step + n.duration)), None)

        if self.active_tool == "delete":
            if clicked_note:
                self.save_undo_state()
                notes.remove(clicked_note)
                self.draw_piano_roll()
                self.update_waveform_display()
            return

        if self.active_tool == "select":
            if clicked_note:
                self.save_undo_state()
                self.selected_note = clicked_note
                self.is_moving_note = True
                self.drag_start_x = x
                self.drag_start_y = y
                self.note_start_step_orig = clicked_note.start_step
                self.note_start_pitch_orig = clicked_note.pitch_idx
                self.preview_note_sound(pitch_idx)
            return

        if self.active_tool == "paint":
            if clicked_note:
                self.selected_note = clicked_note
                note_x2 = (clicked_note.start_step + clicked_note.duration) * self.step_width
                if abs(x - note_x2) <= 10:
                    self.save_undo_state()
                    self.is_resizing = True
                self.drag_start_x = x
                return

            self.save_undo_state()
            new_note = NoteBlock(pitch_idx=pitch_idx, start_step=step, duration_steps=2)
            notes.append(new_note)
            self.check_and_expand_pattern()
            self.preview_note_sound(pitch_idx)
            self.draw_piano_roll()
            self.update_waveform_display()

    def on_piano_roll_drag(self, event):
        x = self.pr_canvas.canvasx(event.x)
        y = self.pr_canvas.canvasy(event.y)

        if self.selected_note and self.is_resizing and self.active_tool == "paint":
            diff_steps = int((x - self.drag_start_x) // self.step_width)
            new_duration = max(1, self.selected_note.duration + diff_steps)
            if new_duration != self.selected_note.duration:
                self.selected_note.duration = new_duration
                self.drag_start_x = x
                self.check_and_expand_pattern()
                self.draw_piano_roll()

        elif self.selected_note and self.is_moving_note and self.active_tool == "select":
            diff_steps = int((x - self.drag_start_x) // self.step_width)
            diff_pitch = int((y - self.drag_start_y) // self.row_height)

            new_step = max(0, self.note_start_step_orig + diff_steps)
            new_pitch = max(0, min(len(ALL_NOTES) - 1, self.note_start_pitch_orig + diff_pitch))

            if new_step != self.selected_note.start_step or new_pitch != self.selected_note.pitch_idx:
                self.selected_note.start_step = new_step
                if new_pitch != self.selected_note.pitch_idx:
                    self.selected_note.pitch_idx = new_pitch
                    self.preview_note_sound(new_pitch)

                self.check_and_expand_pattern()
                self.draw_piano_roll()

    def on_piano_roll_release(self, event):
        self.is_resizing = False
        self.is_moving_note = False

    def on_piano_roll_right_click(self, event):
        if self.active_pattern_idx < 0: return

        x = self.pr_canvas.canvasx(event.x)
        y = self.pr_canvas.canvasy(event.y)

        step = int(x // self.step_width)
        pitch_idx = int(y // self.row_height)

        notes = self.patterns[self.active_pattern_idx].track_notes.get(self.active_track_idx, [])
        for note in list(notes):
            if note.pitch_idx == pitch_idx and note.start_step <= step < (note.start_step + note.duration):
                self.save_undo_state()
                notes.remove(note)
                self.draw_piano_roll()
                self.update_waveform_display()
                break

    def draw_playlist_headers(self):
        self.pl_headers.delete("all")
        total_h = len(self.playlist_tracks) * self.playlist_row_h
        for i in range(len(self.playlist_tracks)):
            y1 = i * self.playlist_row_h
            y2 = y1 + self.playlist_row_h
            self.pl_headers.create_rectangle(0, y1, 100, y2, fill=COLOR_PANEL, outline=COLOR_LINE)
            self.pl_headers.create_text(10, y1 + 20, text=f"Track {i+1}", fill=COLOR_TEXT, anchor="w", font=("Segoe UI", 9, "bold"))

        self.pl_headers.config(scrollregion=(0, 0, 100, total_h))

    def draw_playlist(self):
        self.pl_canvas.delete("all")
        total_w = self.playlist_steps * self.step_width
        total_h = len(self.playlist_tracks) * self.playlist_row_h

        for i in range(len(self.playlist_tracks)):
            y1 = i * self.playlist_row_h
            y2 = y1 + self.playlist_row_h
            self.pl_canvas.create_rectangle(0, y1, total_w, y2, fill=COLOR_GRID_DARK if i % 2 == 0 else COLOR_GRID_LIGHT, outline="")

        for step in range(self.playlist_steps):
            x = step * self.step_width
            is_bar = (step % 16 == 0)
            line_col = COLOR_ACCENT if is_bar else COLOR_LINE
            width = 2 if is_bar else 1
            self.pl_canvas.create_line(x, 0, x, total_h, fill=line_col, width=width)

        total_notes = len(ALL_NOTES)

        for track_idx, clips in enumerate(self.playlist_tracks):
            y1 = track_idx * self.playlist_row_h + 2
            y2 = y1 + self.playlist_row_h - 4

            for clip in clips:
                if clip.pattern_idx >= len(self.patterns): continue

                x1 = clip.start_step * self.step_width
                x2 = x1 + (clip.duration * self.step_width)

                self.pl_canvas.create_rectangle(x1 + 1, y1, x2 - 1, y2, fill=COLOR_PATTERN_CLIP, outline=COLOR_PATTERN_BORDER, width=1)
                self.pl_canvas.create_rectangle(x2 - 6, y1, x2 - 1, y2, fill=COLOR_ACCENT, outline="")

                pat = self.patterns[clip.pattern_idx]
                for tr_idx, notes in pat.track_notes.items():
                    for note in notes:
                        rel_pitch = 1.0 - (note.pitch_idx / total_notes)
                        ny = y1 + 14 + int(rel_pitch * (self.playlist_row_h - 20))

                        nx1 = x1 + (note.start_step * self.step_width)
                        nx2 = nx1 + (note.duration * self.step_width)

                        if nx1 < x2:
                            self.pl_canvas.create_line(nx1, ny, min(nx2, x2), ny, fill="#e67e22", width=2)

                self.pl_canvas.create_text(x1 + 6, y1 + 8, text=pat.name, fill=COLOR_BG, anchor="nw", font=("Segoe UI", 7, "bold"))

        if self.play_mode == "SONG":
            px = self.current_step * self.step_width
            self.pl_canvas.create_line(px, 0, px, total_h, fill=COLOR_PLAYHEAD, width=3)

        self.pl_canvas.config(scrollregion=(0, 0, total_w, total_h))

    def on_playlist_click(self, event):
        x = self.pl_canvas.canvasx(event.x)
        y = self.pl_canvas.canvasy(event.y)

        step = int(x // self.step_width)
        track_idx = int(y // self.playlist_row_h)

        if not (0 <= track_idx < len(self.playlist_tracks) and 0 <= step < self.playlist_steps): return

        clips = self.playlist_tracks[track_idx]
        for clip in clips:
            if clip.start_step <= step < (clip.start_step + clip.duration):
                self.selected_clip = clip
                clip_x2 = (clip.start_step + clip.duration) * self.step_width
                if abs(x - clip_x2) <= 10: self.is_resizing = True
                self.drag_start_x = x
                return

        if self.active_pattern_idx < 0 or not self.patterns: self.add_new_pattern()

        pat = self.patterns[self.active_pattern_idx]
        new_clip = PatternClip(pattern_idx=self.active_pattern_idx, start_step=step, duration_steps=pat.total_steps)
        clips.append(new_clip)
        self.draw_playlist()

    def on_playlist_drag(self, event):
        if self.selected_clip and self.is_resizing:
            x = self.pl_canvas.canvasx(event.x)
            diff_steps = int((x - self.drag_start_x) // self.step_width)
            new_duration = max(2, self.selected_clip.duration + diff_steps)
            if new_duration != self.selected_clip.duration:
                self.selected_clip.duration = new_duration
                self.drag_start_x = x
                self.draw_playlist()

    def on_playlist_release(self, event):
        self.selected_clip = None
        self.is_resizing = False

    def on_playlist_right_click(self, event):
        x = self.pl_canvas.canvasx(event.x)
        y = self.pl_canvas.canvasy(event.y)

        step = int(x // self.step_width)
        track_idx = int(y // self.playlist_row_h)

        if 0 <= track_idx < len(self.playlist_tracks):
            clips = self.playlist_tracks[track_idx]
            for clip in list(clips):
                if clip.start_step <= step < (clip.start_step + clip.duration):
                    clips.remove(clip)
                    self.draw_playlist()
                    break

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.config(bg=COLOR_ACCENT if self.is_playing else COLOR_SURFACE)
        if self.is_playing: self.last_played_step = -1

    def stop_play(self):
        self.is_playing = False
        self.current_step = 0
        self.last_played_step = -1
        self.btn_play.config(bg=COLOR_SURFACE)
        self.draw_piano_roll()
        self.draw_playlist()

if __name__ == "__main__":
    app = LunaBox()
    app.mainloop()
