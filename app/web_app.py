"""
web_app.py: Institutional Venture Capital Underwriting & Precedent Intelligence Terminal.

Clean, professional, window-wise / workflow-guided architecture:
- Window 1: Step-Wise Venture Underwriting Questionnaire (5 Structured Steps, 28 Features)
- Window 2: Executive Investment Memorandum & Precedent Terminal (Tabbed Workbenches)
- Window 3: Fund Deal-Room Capacity Allocator (Top-K Portfolio Construction)
- Interactive Radar Chart (Chart.js), 2D Sensitivity Grid, and Monte Carlo Return Curve
- Zero clutter: spacious, focused views with seamless step transitions

Usage:
    python app/web_app.py --port 8080
"""

import sys
import json
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import argparse
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.precedent.schema import StartupIdea
from src.validator.pipeline import BCAPMValidator
from src.validator.report import format_verdict_report
from src.decision.topk import rank_and_select_top_k
from src.utils import get_logger

logger = get_logger("BCAPM_WebApp")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BCAPM — Institutional Venture Underwriting & Precedent Terminal</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-base: #06080d;
            --bg-surface: #0b0f19;
            --bg-card: rgba(13, 18, 32, 0.88);
            --bg-elevated: rgba(20, 28, 48, 0.75);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-active: rgba(6, 182, 212, 0.5);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-indigo: #6366f1;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-amber: #f59e0b;
            --accent-purple: #a855f7;
            --glow-cyan: 0 0 35px rgba(6, 182, 212, 0.25);
            --glow-emerald: 0 0 35px rgba(16, 185, 129, 0.25);
            --glow-rose: 0 0 35px rgba(244, 63, 94, 0.25);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(6, 182, 212, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 90% 15%, rgba(99, 102, 241, 0.07) 0%, transparent 40%),
                radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%);
        }
        
        /* HEADER */
        header {
            padding: 1rem 2.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-subtle);
            backdrop-filter: blur(20px);
            background: rgba(7, 9, 14, 0.92);
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        .logo-wrap {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .logo-mark {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-indigo));
            color: #fff;
            font-family: 'Outfit', sans-serif;
            font-weight: 900;
            font-size: 1.2rem;
            padding: 0.4rem 0.85rem;
            border-radius: 10px;
            letter-spacing: 0.05em;
            box-shadow: 0 4px 18px rgba(6, 182, 212, 0.35);
        }
        .logo-text h1 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 1.2rem;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .logo-text p {
            font-size: 0.72rem;
            color: var(--text-muted);
            letter-spacing: 0.03em;
        }

        /* TOP WINDOW WORKSPACE SWITCHER */
        .window-nav {
            display: flex;
            background: rgba(255, 255, 255, 0.04);
            padding: 0.3rem;
            border-radius: 12px;
            border: 1px solid var(--border-subtle);
            gap: 0.3rem;
        }
        .window-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 0.5rem 1.2rem;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .window-btn.active {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.25), rgba(99, 102, 241, 0.3));
            color: #fff;
            border: 1px solid var(--accent-cyan);
            box-shadow: 0 2px 10px rgba(6, 182, 212, 0.2);
        }

        .header-stats {
            display: flex;
            align-items: center;
            gap: 1.2rem;
        }
        .stat-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            padding: 0.35rem 0.85rem;
            border-radius: 20px;
            font-size: 0.74rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }
        .pulse-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: var(--accent-emerald);
            box-shadow: 0 0 8px var(--accent-emerald);
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

        /* MAIN CONTAINER */
        main {
            flex: 1;
            width: 100%;
            margin: 0 auto;
            padding: 2.2rem 2.5rem;
        }
        .window-pane {
            display: none;
            width: 100%;
        }
        .window-pane.active {
            display: block;
        }

        /* WINDOW 1: FORM WIZARD VIEW */
        .form-window-wrap {
            max-width: 960px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 1.8rem;
        }
        .card-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 2.2rem;
            backdrop-filter: blur(24px);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        }

        /* PRESETS BAR */
        .presets-wrap {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle);
            padding: 0.85rem 1.4rem;
            border-radius: 14px;
            flex-wrap: wrap;
            gap: 0.8rem;
        }
        .presets-list {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .preset-pill {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            font-size: 0.76rem;
            padding: 0.4rem 0.85rem;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 600;
        }
        .preset-pill:hover {
            background: rgba(6, 182, 212, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }

        /* STEPPER BREADCRUMB HEADER */
        .stepper-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            position: relative;
        }
        .stepper-header::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 5%;
            right: 5%;
            height: 2px;
            background: rgba(255, 255, 255, 0.08);
            z-index: 1;
        }
        .step-item {
            position: relative;
            z-index: 2;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.4rem;
            cursor: pointer;
            background: var(--bg-surface);
            padding: 0 0.8rem;
        }
        .step-circle {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 0.85rem;
            color: var(--text-muted);
            transition: all 0.25s;
        }
        .step-label {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.03em;
            transition: color 0.25s;
        }
        .step-item.active .step-circle {
            background: var(--accent-cyan);
            border-color: var(--accent-cyan);
            color: #000;
            box-shadow: 0 0 16px rgba(6, 182, 212, 0.5);
        }
        .step-item.active .step-label {
            color: #fff;
        }
        .step-item.completed .step-circle {
            background: rgba(16, 185, 129, 0.2);
            border-color: var(--accent-emerald);
            color: var(--accent-emerald);
        }
        .step-item.completed .step-label {
            color: var(--accent-emerald);
        }

        /* STEP PANE CONTAINERS */
        .step-pane {
            display: none;
            flex-direction: column;
            gap: 1.5rem;
            animation: fadeIn 0.3s ease;
        }
        .step-pane.active {
            display: flex;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

        .step-pane-header {
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 1rem;
        }
        .step-pane-header h3 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            color: #fff;
        }
        .step-pane-header p {
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }

        /* FORM INPUTS */
        .form-grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.2rem;
        }
        .form-grid-3 {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1.2rem;
        }
        @media (max-width: 768px) {
            .form-grid-2, .form-grid-3 { grid-template-columns: 1fr; }
        }
        .field-group {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }
        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
        }
        input[type="text"], select, textarea {
            background: rgba(11, 15, 25, 0.9);
            border: 1px solid var(--border-subtle);
            color: #fff;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            font-size: 0.9rem;
            font-family: inherit;
            outline: none;
            transition: all 0.2s;
            width: 100%;
        }
        input[type="text"]:focus, select:focus, textarea:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 16px rgba(6, 182, 212, 0.25);
        }
        textarea {
            resize: vertical;
            min-height: 80px;
        }

        /* SLIDERS */
        .slider-wrap input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 7px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            outline: none;
        }
        .slider-wrap input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--accent-cyan);
            cursor: pointer;
            box-shadow: 0 0 12px var(--accent-cyan);
            transition: transform 0.1s;
        }
        .slider-wrap input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }
        .label-val {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-cyan);
            font-weight: 700;
        }

        /* TOGGLE CARDS */
        .toggles-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.2rem;
        }
        @media (max-width: 600px) {
            .toggles-grid { grid-template-columns: 1fr; }
        }
        .toggle-box {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle);
            padding: 1rem 1.2rem;
            border-radius: 12px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        .switch {
            position: relative;
            display: inline-block;
            width: 44px;
            height: 24px;
            flex-shrink: 0;
        }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider-switch {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(255, 255, 255, 0.1);
            transition: .3s;
            border-radius: 24px;
            border: 1px solid var(--border-subtle);
        }
        .slider-switch:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }
        input:checked + .slider-switch {
            background-color: var(--accent-cyan);
            border-color: var(--accent-cyan);
        }
        input:checked + .slider-switch:before {
            transform: translateX(20px);
        }

        /* STEP FOOTER NAVIGATION */
        .step-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-subtle);
        }
        .btn-prev {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-prev:hover {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }
        .btn-next {
            background: rgba(6, 182, 212, 0.15);
            border: 1px solid var(--accent-cyan);
            color: var(--accent-cyan);
            padding: 0.75rem 1.8rem;
            border-radius: 10px;
            font-size: 0.85rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-next:hover {
            background: var(--accent-cyan);
            color: #000;
            box-shadow: 0 0 16px rgba(6, 182, 212, 0.4);
        }
        .btn-submit-wizard {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            border: none;
            color: #000;
            padding: 0.85rem 2.2rem;
            border-radius: 12px;
            font-size: 0.95rem;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.25s;
            box-shadow: 0 4px 20px rgba(6, 182, 212, 0.4);
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .btn-submit-wizard:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 28px rgba(6, 182, 212, 0.55);
        }

        /* WINDOW 2: INVESTMENT MEMORANDUM & ANALYSIS VIEW */
        .memo-window-wrap {
            max-width: 1480px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 1.8rem;
        }
        
        /* EXECUTIVE HERO BANNER */
        .hero-banner {
            background: rgba(13, 18, 32, 0.9);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 1.8rem 2.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(20px);
        }
        .hero-banner::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 6px;
            background: var(--accent-cyan);
        }
        .hero-left h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .hero-left p {
            font-size: 0.88rem;
            color: var(--text-secondary);
            margin-top: 0.3rem;
        }
        .hero-actions {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .badge-verdict {
            font-family: 'Outfit', sans-serif;
            font-weight: 900;
            font-size: 1.5rem;
            letter-spacing: 0.08em;
            padding: 0.6rem 2rem;
            border-radius: 14px;
            text-transform: uppercase;
        }
        .badge-invest {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            border: 1px solid var(--accent-emerald);
            box-shadow: var(--glow-emerald);
        }
        .badge-review {
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-amber);
            border: 1px solid var(--accent-amber);
        }
        .badge-reject {
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-rose);
            border: 1px solid var(--accent-rose);
            box-shadow: var(--glow-rose);
        }

        /* KPI TILES STRIP */
        .kpi-strip {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.2rem;
        }
        @media (max-width: 900px) {
            .kpi-strip { grid-template-columns: repeat(2, 1fr); }
        }
        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.4rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        .kpi-card-title {
            font-size: 0.74rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
        }
        .kpi-card-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 800;
            margin: 0.35rem 0;
        }
        .kpi-card-sub {
            font-size: 0.76rem;
            color: var(--text-secondary);
        }

        /* DUAL GAUGE */
        .gauge-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.4rem 1.8rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        .gauge-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        .gauge-bar-track {
            position: relative;
            height: 16px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            overflow: visible;
        }
        .gauge-fill {
            height: 100%;
            background: linear-gradient(to right, var(--accent-indigo), var(--accent-cyan));
            border-radius: 10px;
            transition: width 0.6s ease;
        }
        .gauge-hurdle-marker {
            position: absolute;
            top: -6px;
            width: 4px;
            height: 28px;
            background: var(--accent-rose);
            box-shadow: 0 0 10px var(--accent-rose);
            z-index: 2;
        }
        .gauge-footer {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
            margin-top: 0.6rem;
        }

        /* WORKBENCH TAB NAVIGATOR */
        .workbench-nav {
            display: flex;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 0.4rem;
            gap: 0.4rem;
            overflow-x: auto;
        }
        .wb-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 0.65rem 1.3rem;
            border-radius: 10px;
            font-size: 0.84rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            white-space: nowrap;
        }
        .wb-btn.active {
            background: rgba(6, 182, 212, 0.15);
            color: var(--accent-cyan);
            border: 1px solid var(--accent-cyan);
        }

        /* WORKBENCH PANES */
        .wb-pane {
            display: none;
        }
        .wb-pane.active {
            display: block;
        }

        /* WORKBENCH A: RADAR & PRECEDENTS */
        .radar-deck-grid {
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 1.8rem;
        }
        @media (max-width: 1050px) {
            .radar-deck-grid { grid-template-columns: 1fr; }
        }
        .radar-container {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 1.8rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .precedents-scroll {
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
            max-height: 480px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }
        .precedent-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.2rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .precedent-card:hover, .precedent-card.selected {
            border-color: var(--accent-cyan);
            background: rgba(6, 182, 212, 0.05);
        }
        .tag-pill {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            text-transform: uppercase;
        }
        .tag-mna { background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); }
        .tag-closed { background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); }
        .attr-bars-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.5rem;
        }
        .attr-box {
            font-size: 0.68rem;
            color: var(--text-muted);
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .mini-track {
            height: 5px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
            overflow: hidden;
        }
        .mini-fill {
            height: 100%;
            background: var(--accent-cyan);
            border-radius: 3px;
        }

        /* WORKBENCH B: 2D SENSITIVITY GRID */
        .matrix-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin-top: 1rem;
        }
        .matrix-table th {
            text-align: center;
            padding: 0.9rem;
            color: var(--text-muted);
            font-weight: 700;
            border-bottom: 1px solid var(--border-subtle);
        }
        .matrix-table td {
            text-align: center;
            padding: 1.2rem 0.6rem;
            border: 1px solid rgba(255, 255, 255, 0.04);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            transition: all 0.15s;
        }
        .matrix-cell-invest { background: rgba(16, 185, 129, 0.12); color: var(--accent-emerald); font-weight: 700; }
        .matrix-cell-marginal { background: rgba(245, 158, 11, 0.1); color: var(--accent-amber); }
        .matrix-cell-reject { background: rgba(244, 63, 94, 0.1); color: var(--accent-rose); }
        .matrix-active-cell { outline: 2px solid var(--accent-cyan); box-shadow: 0 0 15px rgba(6, 182, 212, 0.4); }

        /* WORKBENCH C: SCENARIOS & POWER LAW */
        .sc-split-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.8rem;
            align-items: center;
        }
        @media (max-width: 900px) {
            .sc-split-grid { grid-template-columns: 1fr; }
        }
        .sc-stat-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.5rem;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.2rem;
        }

        /* WORKBENCH D: MACRO FRICTION */
        .friction-split {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 800px) {
            .friction-split { grid-template-columns: 1fr; }
        }
        .diag-box {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.8rem;
        }
        .diag-box h4 {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .diag-box ul {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
            font-size: 0.88rem;
            color: var(--text-secondary);
        }
        .diag-box ul li {
            position: relative;
            padding-left: 1.6rem;
            line-height: 1.5;
        }
        .diag-box ul li::before {
            position: absolute;
            left: 0;
            font-weight: bold;
        }
        .diag-support ul li::before { content: '✔'; color: var(--accent-emerald); }
        .diag-adverse ul li::before { content: '⚠'; color: var(--accent-rose); }

        /* WORKBENCH E: MARKDOWN MEMO PREVIEW */
        .memo-preview-box {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 2rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.6;
            color: #cbd5e1;
            white-space: pre-wrap;
            max-height: 550px;
            overflow-y: auto;
        }

        /* WINDOW 3: FUND PIPELINE ALLOCATOR */
        .pipeline-window-wrap {
            max-width: 1480px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 1.8rem;
        }
        .pipeline-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }
        .pipeline-table th {
            text-align: left;
            padding: 1rem 1.2rem;
            color: var(--text-muted);
            font-weight: 700;
            border-bottom: 1px solid var(--border-subtle);
        }
        .pipeline-table td {
            padding: 1.2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .row-allocated {
            background: rgba(16, 185, 129, 0.08);
            border-left: 4px solid var(--accent-emerald);
        }
        .row-rejected {
            background: rgba(255, 255, 255, 0.01);
            opacity: 0.65;
        }
        .pipeline-badge {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 0.78rem;
            padding: 0.3rem 0.75rem;
            border-radius: 8px;
            display: inline-block;
        }
        .pipeline-badge-approved { background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); border: 1px solid var(--accent-emerald); }
        .pipeline-badge-passed { background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); }

        /* LOADING OVERLAY */
        #modal-loading {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(6, 8, 13, 0.85);
            backdrop-filter: blur(16px);
            z-index: 2000;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .spinner-ring {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(6, 182, 212, 0.15);
            border-top: 4px solid var(--accent-cyan);
            border-radius: 50%;
            animation: spin 0.75s linear infinite;
            margin-bottom: 1.5rem;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <!-- TOP APP HEADER -->
    <header>
        <div class="logo-wrap">
            <div class="logo-mark">BCAPM</div>
            <div class="logo-text">
                <h1>Backdrop-Conditioned Analogy Precedent Modeling</h1>
                <p>Institutional Venture Underwriting & Capital Decision System</p>
            </div>
        </div>

        <nav class="window-nav">
            <button type="button" class="window-btn active" id="btn-win-form" onclick="switchWindow('form')">
                <span>📝</span> 1. Venture Intake Questionnaire
            </button>
            <button type="button" class="window-btn" id="btn-win-memo" onclick="switchWindow('memo')">
                <span>📊</span> 2. Investment Memorandum & Analysis
            </button>
            <button type="button" class="window-btn" id="btn-win-pipeline" onclick="switchWindow('pipeline')">
                <span>💼</span> 3. Fund Deal-Room Allocator (Top-K)
            </button>
        </nav>

        <div class="header-stats">
            <div class="stat-badge"><span class="pulse-dot"></span> Precedents: 13,258 Deals</div>
            <div class="stat-badge">Asymmetric Loss Engine Active</div>
        </div>
    </header>

    <main>
        <!-- ========================================================= -->
        <!-- WINDOW 1: STEP-WISE VENTURE INTAKE QUESTIONNAIRE          -->
        <!-- ========================================================= -->
        <div id="window-form" class="window-pane active">
            <div class="form-window-wrap">
                <!-- PRESET ARCHETYPES BAR -->
                <div class="presets-wrap">
                    <span style="font-size: 0.78rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">
                        ⚡ Quick Load Verified Archetype:
                    </span>
                    <div class="presets-list">
                        <button type="button" class="preset-pill" onclick="loadPreset('ai_saas')">🤖 AI Health SaaS</button>
                        <button type="button" class="preset-pill" onclick="loadPreset('fintech')">💳 B2B FinTech API</button>
                        <button type="button" class="preset-pill" onclick="loadPreset('d2c')">🛍️ D2C Brand</button>
                        <button type="button" class="preset-pill" onclick="loadPreset('cleantech')">⚡ CleanTech Battery</button>
                        <button type="button" class="preset-pill" onclick="loadPreset('web3')">🔗 Web3 Compute</button>
                    </div>
                </div>

                <div class="card-panel">
                    <!-- PROGRESS STEPPER BREADCRUMBS -->
                    <div class="stepper-header">
                        <div class="step-item active" id="step-nav-1" onclick="jumpToStep(1)">
                            <div class="step-circle">1</div>
                            <span class="step-label">Identity & Cluster</span>
                        </div>
                        <div class="step-item" id="step-nav-2" onclick="jumpToStep(2)">
                            <div class="step-circle">2</div>
                            <span class="step-label">Deal Economics</span>
                        </div>
                        <div class="step-item" id="step-nav-3" onclick="jumpToStep(3)">
                            <div class="step-circle">3</div>
                            <span class="step-label">Founding Team</span>
                        </div>
                        <div class="step-item" id="step-nav-4" onclick="jumpToStep(4)">
                            <div class="step-circle">4</div>
                            <span class="step-label">Product & GTM</span>
                        </div>
                        <div class="step-item" id="step-nav-5" onclick="jumpToStep(5)">
                            <div class="step-circle">5</div>
                            <span class="step-label">Traction & Review</span>
                        </div>
                    </div>

                    <form id="underwriting-form">
                        <!-- STEP 1: IDENTITY & GEOGRAPHY -->
                        <div class="step-pane active" id="step-pane-1">
                            <div class="step-pane-header">
                                <h3>Step 1: Venture Identification & Geographic Cluster</h3>
                                <p>Specify company nomenclature, industrial categorization, and physical innovation hub.</p>
                            </div>
                            <div class="field-group">
                                <label for="startup-name">Opportunity / Venture Title</label>
                                <input type="text" id="startup-name" value="CognitiveHealth AI" required placeholder="e.g. CognitiveHealth AI">
                            </div>
                            <div class="form-grid-2">
                                <div class="field-group">
                                    <label for="market-category">Market Sector Classification</label>
                                    <select id="market-category">
                                        <option value="software">Software / SaaS</option>
                                        <option value="health" selected>Healthcare / MedTech</option>
                                        <option value="biotech">Biotech / Life Sciences</option>
                                        <option value="finance">Fintech / Financial</option>
                                        <option value="ecommerce">E-Commerce / Consumer</option>
                                        <option value="enterprise">Enterprise IT</option>
                                        <option value="hardware">Hardware / CleanTech</option>
                                        <option value="mobile">Mobile / Telecomm</option>
                                        <option value="media">Media / Content</option>
                                    </select>
                                </div>
                                <div class="field-group">
                                    <label for="country-code">National Jurisdiction</label>
                                    <select id="country-code">
                                        <option value="USA" selected>United States (USA)</option>
                                        <option value="GBR">United Kingdom (GBR)</option>
                                        <option value="CAN">Canada (CAN)</option>
                                        <option value="DEU">Germany (DEU)</option>
                                        <option value="IND">India (IND)</option>
                                        <option value="ISR">Israel (ISR)</option>
                                        <option value="FRA">France (FRA)</option>
                                        <option value="SGP">Singapore (SGP)</option>
                                    </select>
                                </div>
                            </div>
                            <div class="form-grid-2">
                                <div class="field-group">
                                    <label for="city-hub">Innovation City Hub / Cluster</label>
                                    <input type="text" id="city-hub" value="San Francisco" placeholder="e.g. San Francisco, London, Berlin, Bengaluru">
                                </div>
                                <div class="field-group">
                                    <label for="founding-year">Founding Era</label>
                                    <select id="founding-year">
                                        <option value="2026" selected>2026 (Current Era)</option>
                                        <option value="2025">2025</option>
                                        <option value="2024">2024</option>
                                        <option value="2022">2022</option>
                                        <option value="2020">2020</option>
                                    </select>
                                </div>
                            </div>
                            <div class="step-footer">
                                <div></div>
                                <button type="button" class="btn-next" onclick="jumpToStep(2)">Continue to Deal Economics →</button>
                            </div>
                        </div>

                        <!-- STEP 2: DEAL ECONOMICS & HURDLE RATE -->
                        <div class="step-pane" id="step-pane-2">
                            <div class="step-pane-header">
                                <h3>Step 2: Capital Economics & Investment Hurdle</h3>
                                <p>Define financing tranche check size and target liquidity payoff to compute the breakeven hurdle rate $p^* = c / V$.</p>
                            </div>
                            <div class="field-group slider-wrap">
                                <label>
                                    <span>Proposed Investment Check Size ($M)</span>
                                    <span class="label-val" id="check-val">$1.75M</span>
                                </label>
                                <input type="range" id="check-size" min="0.25" max="10.0" step="0.25" value="1.75" oninput="updateSliders()">
                            </div>
                            <div class="field-group slider-wrap">
                                <label>
                                    <span>Target Liquidity Exit Payoff ($M)</span>
                                    <span class="label-val" id="exit-val">$35.0M</span>
                                </label>
                                <input type="range" id="target-exit" min="5.0" max="100.0" step="5.0" value="35.0" oninput="updateSliders()">
                            </div>
                            <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 12px; padding: 1rem 1.4rem; display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Derived Breakeven Hurdle Rate:</span>
                                    <p style="font-size: 0.84rem; color: var(--text-secondary); margin-top: 0.2rem;">Minimum success probability required for Expected Monetary Value &gt; $0</p>
                                </div>
                                <span id="hurdle-preview" style="font-family: 'JetBrains Mono'; font-size: 1.5rem; font-weight: 800; color: var(--accent-cyan);">5.0%</span>
                            </div>
                            <div class="form-grid-2">
                                <div class="field-group">
                                    <label for="funding-rounds">Financing Round Stage</label>
                                    <select id="funding-rounds">
                                        <option value="1" selected>Round 1 (Seed / Pre-Seed)</option>
                                        <option value="2">Round 2 (Series A)</option>
                                        <option value="3">Round 3 (Series B)</option>
                                        <option value="4">Round 4+ (Growth Tranche)</option>
                                    </select>
                                </div>
                                <div class="field-group">
                                    <label for="repeat-investors">Repeat Tier-1 Institutional Backers</label>
                                    <select id="repeat-investors">
                                        <option value="0">0 (First Institutional Round)</option>
                                        <option value="1" selected>1 Repeat Institutional VC</option>
                                        <option value="2">2 Repeat Institutional VCs</option>
                                        <option value="3">3+ Repeat Institutional VCs</option>
                                    </select>
                                </div>
                            </div>
                            <div class="step-footer">
                                <button type="button" class="btn-prev" onclick="jumpToStep(1)">← Back</button>
                                <button type="button" class="btn-next" onclick="jumpToStep(3)">Continue to Founding Team →</button>
                            </div>
                        </div>

                        <!-- STEP 3: FOUNDING TEAM HUMAN CAPITAL -->
                        <div class="step-pane" id="step-pane-3">
                            <div class="step-pane-header">
                                <h3>Step 3: Founding Team Human Capital & Pedigree</h3>
                                <p>Underwrite leadership depth, elite technical pedigree, accelerator backing, and team composition.</p>
                            </div>
                            <div class="form-grid-3">
                                <div class="field-group">
                                    <label for="team-size">Co-Founders Count</label>
                                    <select id="team-size">
                                        <option value="1">1 (Solo Founder)</option>
                                        <option value="2" selected>2 Co-Founders</option>
                                        <option value="3">3 Co-Founders</option>
                                        <option value="4">4+ Co-Founders</option>
                                    </select>
                                </div>
                                <div class="field-group">
                                    <label for="female-ratio">Female Founder Ratio</label>
                                    <select id="female-ratio">
                                        <option value="0.0" selected>0%</option>
                                        <option value="0.33">33%</option>
                                        <option value="0.50">50% (Balanced)</option>
                                        <option value="1.0">100%</option>
                                    </select>
                                </div>
                                <div class="field-group">
                                    <label for="senior-leadership">C-Suite / Executive Team</label>
                                    <select id="senior-leadership">
                                        <option value="1">1 Exec</option>
                                        <option value="2" selected>2 Execs</option>
                                        <option value="4">4 Execs</option>
                                        <option value="6">6+ Execs</option>
                                    </select>
                                </div>
                            </div>
                            <div class="toggles-grid">
                                <div class="toggle-box">
                                    <div>
                                        <strong style="color: #fff;">Tier-1 Tech / FAANG Alumni</strong>
                                        <p style="font-size: 0.74rem; color: var(--text-muted); margin-top: 0.15rem;">Founders held senior engineering/product roles at Google, Meta, Apple, Amazon</p>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox" id="top-company" checked>
                                        <span class="slider-switch"></span>
                                    </label>
                                </div>
                                <div class="toggle-box">
                                    <div>
                                        <strong style="color: #fff;">Top-Tier Accelerator Alumni</strong>
                                        <p style="font-size: 0.74rem; color: var(--text-muted); margin-top: 0.15rem;">Graduated from Y Combinator, Techstars, or 500 Global</p>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox" id="accelerator-backer" checked>
                                        <span class="slider-switch"></span>
                                    </label>
                                </div>
                            </div>
                            <div class="step-footer">
                                <button type="button" class="btn-prev" onclick="jumpToStep(2)">← Back</button>
                                <button type="button" class="btn-next" onclick="jumpToStep(4)">Continue to Product Architecture →</button>
                            </div>
                        </div>

                        <!-- STEP 4: PRODUCT ARCHITECTURE & GTM -->
                        <div class="step-pane" id="step-pane-4">
                            <div class="step-pane-header">
                                <h3>Step 4: Product Architecture & Go-To-Market</h3>
                                <p>Formulate business model mechanisms, AI defensibility moat, and target customer profile.</p>
                            </div>
                            <div class="form-grid-2">
                                <div class="field-group">
                                    <label for="business-model">Business Model</label>
                                    <select id="business-model">
                                        <option value="B2B" selected>B2B Enterprise SaaS</option>
                                        <option value="B2C">B2C Direct Consumer</option>
                                        <option value="Hybrid">Hybrid B2B2C Platform</option>
                                    </select>
                                </div>
                                <div class="field-group">
                                    <label for="target-customer">Target Customer Persona</label>
                                    <select id="target-customer">
                                        <option value="Enterprise" selected>Enterprise / Fortune 500</option>
                                        <option value="SMB">Mid-Market SMB</option>
                                        <option value="Consumer">Mass Market Consumer</option>
                                        <option value="Government">Regulated / Government</option>
                                    </select>
                                </div>
                            </div>
                            <div class="toggle-box">
                                <div>
                                    <strong style="color: #fff;">Proprietary Machine Learning / Deep Tech Core</strong>
                                    <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.15rem;">Core value proposition built on trained proprietary models with data flywheel advantage</p>
                                </div>
                                <label class="switch">
                                    <input type="checkbox" id="ml-based" checked>
                                    <span class="slider-switch"></span>
                                </label>
                            </div>
                            <div class="field-group">
                                <label for="venture-description">Executive Investment Thesis & Defensibility</label>
                                <textarea id="venture-description">Enterprise multimodal diagnostics AI platform integrating Electronic Health Records with imaging models to accelerate clinical triage.</textarea>
                            </div>
                            <div class="step-footer">
                                <button type="button" class="btn-prev" onclick="jumpToStep(3)">← Back</button>
                                <button type="button" class="btn-next" onclick="jumpToStep(5)">Continue to Traction & Review →</button>
                            </div>
                        </div>

                        <!-- STEP 5: TRACTION & SUBMISSION -->
                        <div class="step-pane" id="step-pane-5">
                            <div class="step-pane-header">
                                <h3>Step 5: Early Traction, Developer Sentiment & Underwriting Execution</h3>
                                <p>Calibrate community buzz, technical early adopter traction, and execute full precedent matching.</p>
                            </div>
                            <div class="field-group slider-wrap">
                                <label>
                                    <span>Hacker News / Developer Sentiment Score (0.10 to 1.00)</span>
                                    <span class="label-val" id="sentiment-val">0.78</span>
                                </label>
                                <input type="range" id="hn-sentiment" min="0.10" max="1.00" step="0.05" value="0.78" oninput="updateSentimentSliders()">
                            </div>
                            <div class="field-group slider-wrap">
                                <label>
                                    <span>Community Discussion & Engagement Volume Index (0 to 100)</span>
                                    <span class="label-val" id="engagement-val">75.0</span>
                                </label>
                                <input type="range" id="hn-engagement" min="5.0" max="100.0" step="5.0" value="75.0" oninput="updateSentimentSliders()">
                            </div>
                            <div class="step-footer">
                                <button type="button" class="btn-prev" onclick="jumpToStep(4)">← Back</button>
                                <button type="submit" class="btn-submit-wizard">
                                    <span>⚡ Underwrite Venture & Analyze Precedents</span>
                                    <span>→</span>
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>

        <!-- ========================================================= -->
        <!-- WINDOW 2: INVESTMENT MEMORANDUM & PRECEDENT ANALYSIS      -->
        <!-- ========================================================= -->
        <div id="window-memo" class="window-pane">
            <div class="memo-window-wrap">
                <!-- HERO VERDICT BANNER -->
                <div class="hero-banner">
                    <div class="hero-left">
                        <h2 id="memo-title">CognitiveHealth AI</h2>
                        <p id="memo-subtitle">HEALTHCARE • B2B ENTERPRISE • SAN FRANCISCO, USA • 2026 ERA</p>
                    </div>
                    <div class="hero-actions">
                        <button type="button" class="btn-prev" onclick="switchWindow('form')">
                            <span>✏️</span> Edit Intake Parameters
                        </button>
                        <div id="memo-verdict-badge" class="badge-verdict badge-invest">INVEST</div>
                    </div>
                </div>

                <!-- 4-TILE KPI SUMMARY STRIP -->
                <div class="kpi-strip">
                    <div class="kpi-card">
                        <div class="kpi-card-title">Calibrated P(Exit)</div>
                        <div class="kpi-card-num" id="memo-prob" style="color: var(--accent-cyan);">68.8%</div>
                        <div class="kpi-card-sub">Base Exit Rate: 53.4%</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-card-title">Breakeven Hurdle (p*)</div>
                        <div class="kpi-card-num" id="memo-hurdle">5.0%</div>
                        <div class="kpi-card-sub">p* = Check ($1.75M) / Exit ($35M)</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-card-title">Expected Net EMV</div>
                        <div class="kpi-card-num" id="memo-emv" style="color: var(--accent-emerald);">+$22.3M</div>
                        <div class="kpi-card-sub" id="memo-emv-sub">Check: $1.75M | Exit: $35.0M</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-card-title">Analogue Cohort Exit Rate</div>
                        <div class="kpi-card-num" id="memo-exit-rate">80.0%</div>
                        <div class="kpi-card-sub" id="memo-exit-count">4 of 5 Exited (M&A / IPO)</div>
                    </div>
                </div>

                <!-- DUAL GAUGE SPREAD BAR -->
                <div class="gauge-card">
                    <div class="gauge-header">
                        <span>Calibrated Probability vs. Economic Hurdle Rate Spread</span>
                        <span id="memo-gauge-spread" style="color: var(--accent-emerald);">+63.8% Hurdle Spread (High Safety Margin)</span>
                    </div>
                    <div class="gauge-bar-track">
                        <div class="gauge-fill" id="memo-gauge-fill" style="width: 68.8%;"></div>
                        <div class="gauge-hurdle-marker" id="memo-gauge-marker" style="left: 5.0%;" title="Hurdle Rate"></div>
                    </div>
                    <div class="gauge-footer">
                        <span>0% Breakeven Floor</span>
                        <span id="memo-gauge-hurdle-txt">Hurdle: 5.0%</span>
                        <span>100% Guaranteed Success</span>
                    </div>
                </div>

                <!-- WORKBENCH TAB CONTROLS -->
                <nav class="workbench-nav">
                    <button type="button" class="wb-btn active" id="btn-wb-radar" onclick="switchWorkbench('radar')">
                        <span>🏛️</span> Precedents & Attribute Radar
                    </button>
                    <button type="button" class="wb-btn" id="btn-wb-matrix" onclick="switchWorkbench('matrix')">
                        <span>🧮</span> 2D Economic Sensitivity Grid
                    </button>
                    <button type="button" class="wb-btn" id="btn-wb-scenario" onclick="switchWorkbench('scenario')">
                        <span>🎲</span> Scenario Monte Carlo Density
                    </button>
                    <button type="button" class="wb-btn" id="btn-wb-friction" onclick="switchWorkbench('friction')">
                        <span>🌪️</span> Macro Tailwinds & Headwinds
                    </button>
                    <button type="button" class="wb-btn" id="btn-wb-report" onclick="switchWorkbench('report')">
                        <span>📋</span> Full Memorandum (Markdown)
                    </button>
                </nav>

                <!-- WORKBENCH PANE A: RADAR & PRECEDENT DECK -->
                <div id="wb-pane-radar" class="wb-pane active">
                    <div class="radar-deck-grid">
                        <div class="radar-container">
                            <h4 style="font-size: 0.92rem; font-weight: 700; color: #fff; margin-bottom: 0.5rem;">Decomposed Multi-Attribute Radar</h4>
                            <div style="width: 100%; height: 280px; position: relative;">
                                <canvas id="wbRadarCanvas"></canvas>
                            </div>
                            <span id="wb-radar-caption" style="font-size: 0.74rem; color: var(--text-muted); margin-top: 0.5rem;">
                                Target Startup vs Historical Analogue
                            </span>
                        </div>
                        <div class="card-panel" style="padding: 1.4rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.6rem;">
                                <h3 style="font-family: 'Outfit'; font-size: 1.15rem; color: #fff;">Top Historical Precedents (Pool of 13,258)</h3>
                                <span style="font-size: 0.74rem; color: var(--text-muted);">Click card to benchmark radar</span>
                            </div>
                            <div class="precedents-scroll" id="wb-precedents-deck"></div>
                        </div>
                    </div>
                </div>

                <!-- WORKBENCH PANE B: 2D SENSITIVITY HEATMAP -->
                <div id="wb-pane-matrix" class="wb-pane">
                    <div class="card-panel">
                        <div style="border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.8rem; margin-bottom: 1rem;">
                            <h3 style="font-family: 'Outfit'; font-size: 1.25rem; color: #fff;">2D Economic Sensitivity Heatmap Grid</h3>
                            <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 0.2rem;">
                                Computes Net EMV and breakeven feasibility across 16 Check Size vs Liquidity Valuation exit permutations given calibrated P(Exit).
                            </p>
                        </div>
                        <table class="matrix-table" id="wb-sensitivity-table"></table>
                    </div>
                </div>

                <!-- WORKBENCH PANE C: SCENARIOS & POWER LAW -->
                <div id="wb-pane-scenario" class="wb-pane">
                    <div class="card-panel">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.8rem; margin-bottom: 1.2rem;">
                            <div>
                                <h3 style="font-family: 'Outfit'; font-size: 1.25rem; color: #fff;">Venture Return Scenario Sensitivity</h3>
                                <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 0.2rem;">Simulates 1,000 venture trajectories under heavy-tailed power-law distribution.</p>
                            </div>
                            <div style="display: flex; gap: 0.4rem;">
                                <button type="button" class="preset-pill" id="sc-btn-base" onclick="switchScenario('base')">Base Regime (5x–15x)</button>
                                <button type="button" class="preset-pill" id="sc-btn-cons" onclick="switchScenario('conservative')">Conservative (2x–5x)</button>
                                <button type="button" class="preset-pill" id="sc-btn-aggr" onclick="switchScenario('aggressive')">Aggressive (25x–100x+)</button>
                            </div>
                        </div>
                        <div class="sc-split-grid">
                            <div class="sc-stat-box">
                                <div>
                                    <span style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Mean Return Multiple</span>
                                    <div id="wb-sc-mean" style="font-family: 'JetBrains Mono'; font-size: 1.8rem; font-weight: 800; color: var(--accent-cyan); margin-top: 0.2rem;">6.17x</div>
                                </div>
                                <div>
                                    <span style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Expected Net Return</span>
                                    <div id="wb-sc-net" style="font-family: 'JetBrains Mono'; font-size: 1.8rem; font-weight: 800; color: var(--accent-emerald); margin-top: 0.2rem;">+$9.25M</div>
                                </div>
                                <div>
                                    <span style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">P(Capital Loss)</span>
                                    <div id="wb-sc-loss" style="font-family: 'JetBrains Mono'; font-size: 1.8rem; font-weight: 800; color: var(--accent-rose); margin-top: 0.2rem;">32%</div>
                                </div>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); border-radius: 14px; border: 1px solid var(--border-subtle); padding: 1.2rem;">
                                <span style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Simulated Power-Law Density Curve:</span>
                                <div style="height: 140px; width: 100%; margin-top: 0.5rem;">
                                    <canvas id="wbMonteCarloCanvas"></canvas>
                                </div>
                            </div>
                        </div>
                        <p id="wb-sc-desc" style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 1rem;">
                            Standard historical venture capital power-law distribution (5x–15x median).
                        </p>
                    </div>
                </div>

                <!-- WORKBENCH PANE D: MACRO FRICTION -->
                <div id="wb-pane-friction" class="wb-pane">
                    <div class="friction-split">
                        <div class="diag-box">
                            <h4 style="color: var(--accent-emerald);"><span>✔</span> Supporting Macroeconomic Tailwinds</h4>
                            <ul id="wb-support-list"></ul>
                        </div>
                        <div class="diag-box">
                            <h4 style="color: var(--accent-rose);"><span>⚠</span> Adverse Structural Headwinds</h4>
                            <ul id="wb-adverse-list"></ul>
                        </div>
                    </div>
                </div>

                <!-- WORKBENCH PANE E: MARKDOWN MEMO -->
                <div id="wb-pane-report" class="wb-pane">
                    <div class="card-panel">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.8rem; margin-bottom: 1rem;">
                            <h3 style="font-family: 'Outfit'; font-size: 1.25rem; color: #fff;">Institutional 13-Section Due Diligence Memorandum</h3>
                            <button type="button" class="btn-prev" onclick="copyMarkdownReport()">
                                <span>📋</span> Copy Full Markdown to Clipboard
                            </button>
                        </div>
                        <div class="memo-preview-box" id="wb-memo-markdown-text">Awaiting report generation...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ========================================================= -->
        <!-- WINDOW 3: FUND DEAL-ROOM ALLOCATOR (TOP-K PIPELINE)       -->
        <!-- ========================================================= -->
        <div id="window-pipeline" class="window-pane">
            <div class="pipeline-window-wrap">
                <div class="card-panel">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-subtle); padding-bottom: 1.2rem; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem;">
                        <div>
                            <h2 style="font-family: 'Outfit'; font-size: 1.5rem; color: #fff;">Fund Deal-Room Allocator — Capacity-Constrained Top-K Selection</h2>
                            <p style="font-size: 0.84rem; color: var(--text-muted); margin-top: 0.2rem;">
                                Solves optimal capital deployment across multiple competing opportunities subject to finite fund budget and deal capacity ceiling.
                            </p>
                        </div>
                        <div style="display: flex; gap: 1rem; align-items: center;">
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <label style="font-size: 0.78rem;">Fund Budget:</label>
                                <select id="fund-budget-select" onchange="runPipelineAllocation()" style="padding: 0.5rem 0.8rem; font-size: 0.82rem; width: auto;">
                                    <option value="3.5">$3.5M Seed Pool</option>
                                    <option value="5.0" selected>$5.0M Seed Pool</option>
                                    <option value="7.5">$7.5M Seed Pool</option>
                                    <option value="10.0">$10.0M Seed Pool</option>
                                </select>
                            </div>
                            <div style="display: flex; align-items: center; gap: 0.5rem;">
                                <label style="font-size: 0.78rem;">Capacity (K):</label>
                                <select id="fund-capacity-select" onchange="runPipelineAllocation()" style="padding: 0.5rem 0.8rem; font-size: 0.82rem; width: auto;">
                                    <option value="1">K = 1 Deal</option>
                                    <option value="2" selected>K = 2 Deals</option>
                                    <option value="3">K = 3 Deals</option>
                                    <option value="4">K = 4 Deals</option>
                                </select>
                            </div>
                            <button type="button" class="btn-next" onclick="runPipelineAllocation()">
                                <span>⚡ Re-Optimize Allocations</span>
                            </button>
                        </div>
                    </div>

                    <!-- 4 PIPELINE KPIS -->
                    <div class="kpi-strip" style="margin-bottom: 1.8rem;">
                        <div class="kpi-card">
                            <div class="kpi-card-title">Capital Deployed / Budget</div>
                            <div class="kpi-card-num" id="pipe-capital" style="color: var(--accent-cyan);">$4.25M / $5.0M</div>
                            <div class="kpi-card-sub" id="pipe-unallocated">Unallocated Reserve: $0.75M</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-card-title">Portfolio Total Net EMV</div>
                            <div class="kpi-card-num" id="pipe-emv" style="color: var(--accent-emerald);">+$43.5M</div>
                            <div class="kpi-card-sub">Aggregated portfolio expected upside</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-card-title">Mean Selected P(Exit)</div>
                            <div class="kpi-card-num" id="pipe-prob" style="color: var(--accent-indigo);">70.5%</div>
                            <div class="kpi-card-sub">High-conviction cohort conviction</div>
                        </div>
                        <div class="kpi-card">
                            <div class="kpi-card-title">Expected Fund MOIC</div>
                            <div class="kpi-card-num" id="pipe-moic" style="color: var(--accent-amber);">10.2x</div>
                            <div class="kpi-card-sub">Multiple on deployed capital</div>
                        </div>
                    </div>

                    <!-- PIPELINE DEALS TABLE -->
                    <div style="background: rgba(0,0,0,0.3); border-radius: 16px; border: 1px solid var(--border-subtle); overflow: hidden; margin-bottom: 1.8rem;">
                        <table class="pipeline-table">
                            <thead>
                                <tr>
                                    <th>Allocation Status</th>
                                    <th>Candidate Startup</th>
                                    <th>Sector / Model</th>
                                    <th>Check Size</th>
                                    <th>Target Exit</th>
                                    <th>Calibrated P(Exit)</th>
                                    <th>Hurdle (p*)</th>
                                    <th>Expected Net EMV</th>
                                </tr>
                            </thead>
                            <tbody id="pipeline-tbody"></tbody>
                        </table>
                    </div>

                    <!-- COMPARATIVE BAR CHART -->
                    <div style="background: rgba(0,0,0,0.3); border-radius: 16px; border: 1px solid var(--border-subtle); padding: 1.5rem;">
                        <h4 style="font-size: 0.95rem; font-weight: 700; color: #fff; margin-bottom: 1rem;">Deal Expected Net Monetary Value vs Required Check Size</h4>
                        <div style="height: 240px; width: 100%;">
                            <canvas id="pipelineBarCanvas"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- FULLSCREEN LOADING MODAL -->
    <div id="modal-loading">
        <div class="spinner-ring"></div>
        <h3 style="font-family: 'Outfit'; font-size: 1.5rem; color: #fff;">Underwriting Against 13,258 Historical Deals</h3>
        <p style="font-size: 0.9rem; color: var(--text-muted); margin-top: 0.4rem;">Calibrating multi-attribute similarity matrix and economic hurdle rates...</p>
    </div>

    <script>
        let currentVerdictData = null;
        let activeStep = 1;
        let activeScenarioKey = 'base';
        let radarChartInstance = null;
        let monteCarloChartInstance = null;
        let pipelineBarChartInstance = null;

        // Pipeline candidate deals pool
        const PIPELINE_DEALS = [
            {
                name: 'CognitiveHealth AI',
                market_category: 'health',
                business_model: 'B2B',
                country_code: 'USA',
                city: 'San Francisco',
                founder_count: 2,
                worked_in_top_companies: true,
                cax_cofounders: true,
                repeat_investor_count: 2,
                is_ml_based: true,
                check_size_m: 1.75,
                target_exit_m: 35.0
            },
            {
                name: 'PrimeLedger API',
                market_category: 'finance',
                business_model: 'B2B',
                country_code: 'GBR',
                city: 'London',
                founder_count: 3,
                worked_in_top_companies: true,
                cax_cofounders: false,
                repeat_investor_count: 1,
                is_ml_based: false,
                check_size_m: 2.50,
                target_exit_m: 50.0
            },
            {
                name: 'GlowOrganics Direct',
                market_category: 'ecommerce',
                business_model: 'B2C',
                country_code: 'USA',
                city: 'Los Angeles',
                founder_count: 1,
                worked_in_top_companies: false,
                cax_cofounders: false,
                repeat_investor_count: 0,
                is_ml_based: false,
                check_size_m: 0.50,
                target_exit_m: 10.0
            },
            {
                name: 'Volterra Storage',
                market_category: 'hardware',
                business_model: 'B2B',
                country_code: 'DEU',
                city: 'Berlin',
                founder_count: 3,
                worked_in_top_companies: true,
                cax_cofounders: true,
                repeat_investor_count: 2,
                is_ml_based: true,
                check_size_m: 3.00,
                target_exit_m: 40.0
            },
            {
                name: 'NexusMesh Web3',
                market_category: 'software',
                business_model: 'B2B',
                country_code: 'USA',
                city: 'New York',
                founder_count: 2,
                worked_in_top_companies: true,
                cax_cofounders: true,
                repeat_investor_count: 1,
                is_ml_based: true,
                check_size_m: 1.50,
                target_exit_m: 20.0
            }
        ];

        function switchWindow(target) {
            document.querySelectorAll('.window-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.window-pane').forEach(p => p.classList.remove('active'));

            if (target === 'form') {
                document.getElementById('btn-win-form').classList.add('active');
                document.getElementById('window-form').classList.add('active');
            } else if (target === 'memo') {
                document.getElementById('btn-win-memo').classList.add('active');
                document.getElementById('window-memo').classList.add('active');
            } else if (target === 'pipeline') {
                document.getElementById('btn-win-pipeline').classList.add('active');
                document.getElementById('window-pipeline').classList.add('active');
                runPipelineAllocation();
            }
        }

        function jumpToStep(stepNum) {
            activeStep = stepNum;
            for (let i = 1; i <= 5; i++) {
                const nav = document.getElementById('step-nav-' + i);
                const pane = document.getElementById('step-pane-' + i);
                nav.classList.remove('active', 'completed');
                pane.classList.remove('active');

                if (i === stepNum) {
                    nav.classList.add('active');
                    pane.classList.add('active');
                } else if (i < stepNum) {
                    nav.classList.add('completed');
                }
            }
        }

        function switchWorkbench(wbKey) {
            document.querySelectorAll('.wb-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.wb-pane').forEach(p => p.classList.remove('active'));

            const activeBtn = document.getElementById('btn-wb-' + wbKey);
            const activePane = document.getElementById('wb-pane-' + wbKey);
            if (activeBtn) activeBtn.classList.add('active');
            if (activePane) activePane.classList.add('active');

            if (wbKey === 'radar' && currentVerdictData) {
                initRadarChart(currentVerdictData.retrieval_result.precedents[0]);
            } else if (wbKey === 'scenario') {
                renderMonteCarloChart(activeScenarioKey);
            }
        }

        function updateSliders() {
            const check = parseFloat(document.getElementById('check-size').value);
            const exit = parseFloat(document.getElementById('target-exit').value);
            document.getElementById('check-val').innerText = '$' + check.toFixed(2) + 'M';
            document.getElementById('exit-val').innerText = '$' + exit.toFixed(1) + 'M';
            const hurdle = (check / exit) * 100;
            document.getElementById('hurdle-preview').innerText = hurdle.toFixed(1) + '%';
        }

        function updateSentimentSliders() {
            const sent = parseFloat(document.getElementById('hn-sentiment').value);
            const eng = parseFloat(document.getElementById('hn-engagement').value);
            document.getElementById('sentiment-val').innerText = sent.toFixed(2);
            document.getElementById('engagement-val').innerText = eng.toFixed(1);
        }

        function loadPreset(archetype) {
            if (archetype === 'ai_saas') {
                document.getElementById('startup-name').value = 'CognitiveHealth AI';
                document.getElementById('market-category').value = 'health';
                document.getElementById('business-model').value = 'B2B';
                document.getElementById('country-code').value = 'USA';
                document.getElementById('city-hub').value = 'San Francisco';
                document.getElementById('founding-year').value = '2026';
                document.getElementById('check-size').value = 1.75;
                document.getElementById('target-exit').value = 35.0;
                document.getElementById('funding-rounds').value = '1';
                document.getElementById('repeat-investors').value = '2';
                document.getElementById('team-size').value = '2';
                document.getElementById('female-ratio').value = '0.50';
                document.getElementById('senior-leadership').value = '4';
                document.getElementById('top-company').checked = true;
                document.getElementById('accelerator-backer').checked = true;
                document.getElementById('target-customer').value = 'Enterprise';
                document.getElementById('ml-based').checked = true;
                document.getElementById('venture-description').value = 'Enterprise multimodal diagnostics AI platform integrating Electronic Health Records with imaging models to accelerate clinical triage.';
                document.getElementById('hn-sentiment').value = 0.78;
                document.getElementById('hn-engagement').value = 75.0;
            } else if (archetype === 'fintech') {
                document.getElementById('startup-name').value = 'PrimeLedger API';
                document.getElementById('market-category').value = 'finance';
                document.getElementById('business-model').value = 'B2B';
                document.getElementById('country-code').value = 'GBR';
                document.getElementById('city-hub').value = 'London';
                document.getElementById('founding-year').value = '2025';
                document.getElementById('check-size').value = 2.50;
                document.getElementById('target-exit').value = 50.0;
                document.getElementById('funding-rounds').value = '2';
                document.getElementById('repeat-investors').value = '1';
                document.getElementById('team-size').value = '3';
                document.getElementById('female-ratio').value = '0.33';
                document.getElementById('senior-leadership').value = '4';
                document.getElementById('top-company').checked = true;
                document.getElementById('accelerator-backer').checked = false;
                document.getElementById('target-customer').value = 'Enterprise';
                document.getElementById('ml-based').checked = false;
                document.getElementById('venture-description').value = 'Next-generation cross-border B2B clearing and liquidity settlement API for tier-2 banks and fintechs.';
                document.getElementById('hn-sentiment').value = 0.68;
                document.getElementById('hn-engagement').value = 60.0;
            } else if (archetype === 'd2c') {
                document.getElementById('startup-name').value = 'GlowOrganics Direct';
                document.getElementById('market-category').value = 'ecommerce';
                document.getElementById('business-model').value = 'B2C';
                document.getElementById('country-code').value = 'USA';
                document.getElementById('city-hub').value = 'Los Angeles';
                document.getElementById('founding-year').value = '2026';
                document.getElementById('check-size').value = 0.50;
                document.getElementById('target-exit').value = 10.0;
                document.getElementById('funding-rounds').value = '1';
                document.getElementById('repeat-investors').value = '0';
                document.getElementById('team-size').value = '1';
                document.getElementById('female-ratio').value = '1.0';
                document.getElementById('senior-leadership').value = '1';
                document.getElementById('top-company').checked = false;
                document.getElementById('accelerator-backer').checked = false;
                document.getElementById('target-customer').value = 'Consumer';
                document.getElementById('ml-based').checked = false;
                document.getElementById('venture-description').value = 'Direct-to-consumer organic skincare formulated from plant-based extracts with zero synthetic preservatives.';
                document.getElementById('hn-sentiment').value = 0.52;
                document.getElementById('hn-engagement').value = 35.0;
            } else if (archetype === 'cleantech') {
                document.getElementById('startup-name').value = 'Volterra Storage';
                document.getElementById('market-category').value = 'hardware';
                document.getElementById('business-model').value = 'B2B';
                document.getElementById('country-code').value = 'DEU';
                document.getElementById('city-hub').value = 'Berlin';
                document.getElementById('founding-year').value = '2025';
                document.getElementById('check-size').value = 3.00;
                document.getElementById('target-exit').value = 40.0;
                document.getElementById('funding-rounds').value = '2';
                document.getElementById('repeat-investors').value = '2';
                document.getElementById('team-size').value = '3';
                document.getElementById('female-ratio').value = '0.33';
                document.getElementById('senior-leadership').value = '4';
                document.getElementById('top-company').checked = true;
                document.getElementById('accelerator-backer').checked = true;
                document.getElementById('target-customer').value = 'Enterprise';
                document.getElementById('ml-based').checked = true;
                document.getElementById('venture-description').value = 'Grid-scale solid-state iron battery cells with 4x energy density and 20-year cycle lifetimes.';
                document.getElementById('hn-sentiment').value = 0.74;
                document.getElementById('hn-engagement').value = 80.0;
            } else if (archetype === 'web3') {
                document.getElementById('startup-name').value = 'NexusMesh Compute';
                document.getElementById('market-category').value = 'software';
                document.getElementById('business-model').value = 'B2B';
                document.getElementById('country-code').value = 'USA';
                document.getElementById('city-hub').value = 'New York';
                document.getElementById('founding-year').value = '2026';
                document.getElementById('check-size').value = 1.50;
                document.getElementById('target-exit').value = 20.0;
                document.getElementById('funding-rounds').value = '1';
                document.getElementById('repeat-investors').value = '1';
                document.getElementById('team-size').value = '2';
                document.getElementById('female-ratio').value = '0.0';
                document.getElementById('senior-leadership').value = '2';
                document.getElementById('top-company').checked = true;
                document.getElementById('accelerator-backer').checked = true;
                document.getElementById('target-customer').value = 'Enterprise';
                document.getElementById('ml-based').checked = true;
                document.getElementById('venture-description').value = 'Decentralized peer-to-peer compute network aggregating idle GPU clusters for low-latency AI inference.';
                document.getElementById('hn-sentiment').value = 0.65;
                document.getElementById('hn-engagement').value = 55.0;
            }
            updateSliders();
            updateSentimentSliders();
        }

        document.getElementById('underwriting-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            document.getElementById('modal-loading').style.display = 'flex';

            const payload = {
                name: document.getElementById('startup-name').value,
                market_category: document.getElementById('market-category').value,
                business_model: document.getElementById('business-model').value,
                country_code: document.getElementById('country-code').value,
                city: document.getElementById('city-hub').value,
                founding_year: parseInt(document.getElementById('founding-year').value),
                initial_funding_usd: parseFloat(document.getElementById('check-size').value) * 1000000.0,
                funding_rounds_count: parseInt(document.getElementById('funding-rounds').value),
                repeat_investor_count: parseInt(document.getElementById('repeat-investors').value),
                founder_count: parseInt(document.getElementById('team-size').value),
                female_founder_ratio: parseFloat(document.getElementById('female-ratio').value),
                team_senior_leadership_size: parseInt(document.getElementById('senior-leadership').value),
                worked_in_top_companies: document.getElementById('top-company').checked,
                cax_cofounders: document.getElementById('accelerator-backer').checked,
                is_ml_based: document.getElementById('ml-based').checked,
                target_customer: document.getElementById('target-customer').value,
                description: document.getElementById('venture-description').value,
                hn_sentiment_score: parseFloat(document.getElementById('hn-sentiment').value),
                hn_public_engagement: parseFloat(document.getElementById('hn-engagement').value),
                target_exit_m: parseFloat(document.getElementById('target-exit').value),
                check_size_m: parseFloat(document.getElementById('check-size').value)
            };

            try {
                const response = await fetch('/api/validate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                currentVerdictData = data;
                renderVerdict(data);
                switchWindow('memo');
            } catch (err) {
                alert('Evaluation error: ' + err.message);
            } finally {
                document.getElementById('modal-loading').style.display = 'none';
            }
        });

        function renderVerdict(data) {
            document.getElementById('memo-title').innerText = data.idea.name;
            const cityText = data.idea.city ? `${data.idea.city}, ` : '';
            document.getElementById('memo-subtitle').innerText = `${data.idea.market_category.toUpperCase()} • ${data.idea.business_model} • ${cityText}${data.idea.country_code} • ${data.backdrop_evaluation.backdrop_alignment} MACRO ALIGNMENT`;

            const badge = document.getElementById('memo-verdict-badge');
            badge.innerText = data.verdict;
            badge.className = 'badge-verdict ' + (data.verdict === 'INVEST' ? 'badge-invest' : (data.verdict === 'REVIEW' ? 'badge-review' : 'badge-reject'));

            // KPIs
            const probPct = (data.calibrated_probability * 100).toFixed(1);
            const hurdlePct = (data.economic_threshold * 100).toFixed(1);
            document.getElementById('memo-prob').innerText = probPct + '%';
            document.getElementById('memo-hurdle').innerText = hurdlePct + '%';
            document.getElementById('memo-emv').innerText = (data.expected_monetary_value_m >= 0 ? '+$' : '-$') + Math.abs(data.expected_monetary_value_m).toFixed(2) + 'M';
            document.getElementById('memo-emv-sub').innerText = `Check: $${data.check_size_m.toFixed(2)}M | Exit: $${data.target_exit_m.toFixed(1)}M`;
            document.getElementById('memo-exit-rate').innerText = data.retrieval_result.empirical_exit_percentage;
            document.getElementById('memo-exit-count').innerText = `${data.retrieval_result.exits_count} of ${data.retrieval_result.top_k} Exited (M&A/IPO)`;

            // Dual Gauge
            document.getElementById('memo-gauge-fill').style.width = probPct + '%';
            document.getElementById('memo-gauge-marker').style.left = Math.min(98, Math.max(2, parseFloat(hurdlePct))) + '%';
            document.getElementById('memo-gauge-hurdle-txt').innerText = 'Hurdle: ' + hurdlePct + '%';
            const spread = (data.calibrated_probability - data.economic_threshold) * 100;
            const spreadLabel = document.getElementById('memo-gauge-spread');
            spreadLabel.innerText = (spread >= 0 ? '+' : '') + spread.toFixed(1) + '% Hurdle Spread (' + (spread >= 0 ? 'Positive Safety Margin' : 'Negative EMV') + ')';
            spreadLabel.style.color = spread >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';

            // Render Precedent Deck in Workbench A
            const deck = document.getElementById('wb-precedents-deck');
            deck.innerHTML = '';
            data.retrieval_result.precedents.forEach((p, idx) => {
                const tagClass = p.is_exit === 1 ? 'tag-mna' : 'tag-closed';
                const tagText = p.is_exit === 1 ? 'M&A / IPO Exit' : 'Closed / Liquidation';
                const comps = p.component_attributions || {};

                const card = document.createElement('div');
                card.className = 'precedent-card' + (idx === 0 ? ' selected' : '');
                card.onclick = () => selectPrecedent(idx);
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 700; font-size: 0.95rem; color: #fff;">${p.company_name}</span>
                            <div style="font-size: 0.76rem; color: var(--text-muted); margin-top: 0.15rem;">${p.market_category.toUpperCase()} • ${p.country_code} • Founded ${p.founded_year}</div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.8rem;">
                            <span class="tag-pill ${tagClass}">${tagText}</span>
                            <div style="font-family: 'JetBrains Mono'; font-weight: 700; color: var(--accent-cyan); font-size: 0.9rem;">
                                ${p.similarity_percentage} <span style="font-size: 0.7rem; color: var(--text-muted);">match</span>
                            </div>
                        </div>
                    </div>
                    <div class="attr-bars-grid">
                        <div class="attr-box">
                            <span>Sector</span>
                            <div class="mini-track"><div class="mini-fill" style="width: ${(comps.industry || 0.8) * 100}%;"></div></div>
                        </div>
                        <div class="attr-box">
                            <span>Model</span>
                            <div class="mini-track"><div class="mini-fill" style="width: ${(comps.business_model || 0.8) * 100}%;"></div></div>
                        </div>
                        <div class="attr-box">
                            <span>Geog</span>
                            <div class="mini-track"><div class="mini-fill" style="width: ${(comps.geography || 0.8) * 100}%;"></div></div>
                        </div>
                        <div class="attr-box">
                            <span>Capital</span>
                            <div class="mini-track"><div class="mini-fill" style="width: ${(comps.capital_scale || 0.8) * 100}%;"></div></div>
                        </div>
                        <div class="attr-box">
                            <span>Team</span>
                            <div class="mini-track"><div class="mini-fill" style="width: ${(comps.team || 0.8) * 100}%;"></div></div>
                        </div>
                    </div>
                `;
                deck.appendChild(card);
            });

            // Initialize Radar Chart with Top Precedent
            initRadarChart(data.retrieval_result.precedents[0]);

            // Render 2D Sensitivity Heatmap
            renderSensitivityMatrix(data.calibrated_probability, data.check_size_m, data.target_exit_m);

            // Friction Lists
            const suppList = document.getElementById('wb-support-list');
            suppList.innerHTML = '';
            data.condition_diagnostics.supporting_conditions.forEach(s => suppList.innerHTML += `<li>${s}</li>`);

            const advList = document.getElementById('wb-adverse-list');
            advList.innerHTML = '';
            if (data.condition_diagnostics.adverse_conditions.length === 0) {
                advList.innerHTML = '<li>No significant structural headwinds detected.</li>';
            } else {
                data.condition_diagnostics.adverse_conditions.forEach(a => advList.innerHTML += `<li>${a}</li>`);
            }

            // Scenarios
            switchScenario(activeScenarioKey);

            // Fetch Markdown report preview
            loadMarkdownPreview();
        }

        function selectPrecedent(idx) {
            if (!currentVerdictData) return;
            const cards = document.querySelectorAll('.precedent-card');
            cards.forEach((c, i) => {
                if (i === idx) c.classList.add('selected');
                else c.classList.remove('selected');
            });
            const p = currentVerdictData.retrieval_result.precedents[idx];
            initRadarChart(p);
            document.getElementById('wb-radar-caption').innerText = `Target (${currentVerdictData.idea.name}) vs ${p.company_name} (${p.similarity_percentage})`;
        }

        function initRadarChart(precedent) {
            const ctx = document.getElementById('wbRadarCanvas').getContext('2d');
            const comps = precedent.component_attributions || {};

            const labels = ['Sector', 'Business Model', 'Geography', 'Capital Scale', 'Founder Pedigree'];
            const targetData = [100, 100, 100, 100, 100];
            const precedentData = [
                Math.round((comps.industry || 0.85) * 100),
                Math.round((comps.business_model || 0.8) * 100),
                Math.round((comps.geography || 0.9) * 100),
                Math.round((comps.capital_scale || 0.75) * 100),
                Math.round((comps.team || 0.8) * 100)
            ];

            if (radarChartInstance) {
                radarChartInstance.destroy();
            }

            radarChartInstance = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: currentVerdictData.idea.name + ' (Target)',
                            data: targetData,
                            borderColor: '#06b6d4',
                            backgroundColor: 'rgba(6, 182, 212, 0.2)',
                            borderWidth: 2,
                            pointBackgroundColor: '#06b6d4'
                        },
                        {
                            label: precedent.company_name + ' (Analogue)',
                            data: precedentData,
                            borderColor: precedent.is_exit === 1 ? '#10b981' : '#f43f5e',
                            backgroundColor: precedent.is_exit === 1 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                            borderWidth: 2,
                            pointBackgroundColor: precedent.is_exit === 1 ? '#10b981' : '#f43f5e'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                            grid: { color: 'rgba(255, 255, 255, 0.08)' },
                            pointLabels: {
                                color: '#94a3b8',
                                font: { size: 10, family: 'Plus Jakarta Sans', weight: '600' }
                            },
                            ticks: { display: false, min: 0, max: 100 }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#f8fafc', font: { size: 11 } }
                        }
                    }
                }
            });
        }

        function renderSensitivityMatrix(p, currentCheck, currentExit) {
            const checks = [0.5, 1.0, 2.0, 3.0];
            const exits = [10.0, 25.0, 50.0, 100.0];

            let tableHTML = '<thead><tr><th>Check Size</th>';
            exits.forEach(e => { tableHTML += `<th>$${e}M Exit</th>`; });
            tableHTML += '</tr></thead><tbody>';

            checks.forEach(c => {
                tableHTML += `<tr><td style="font-weight: 700; color: #94a3b8;">$${c.toFixed(1)}M Check</td>`;
                exits.forEach(e => {
                    const hurdle = c / e;
                    const emv = (p * e) - c;
                    const isCurrent = (Math.abs(c - currentCheck) < 0.3 && Math.abs(e - currentExit) < 6.0);
                    
                    let cellClass = 'matrix-cell-reject';
                    if (emv > 2.0) cellClass = 'matrix-cell-invest';
                    else if (emv >= 0) cellClass = 'matrix-cell-marginal';

                    if (isCurrent) cellClass += ' matrix-active-cell';

                    tableHTML += `
                        <td class="${cellClass}" title="Hurdle: ${(hurdle*100).toFixed(1)}% | EMV: $${emv.toFixed(2)}M">
                            ${emv >= 0 ? '+$' : '-$'}${Math.abs(emv).toFixed(1)}M<br>
                            <span style="font-size: 0.7rem; opacity: 0.85;">p* ${(hurdle*100).toFixed(0)}%</span>
                        </td>
                    `;
                });
                tableHTML += '</tr>';
            });
            tableHTML += '</tbody>';
            document.getElementById('wb-sensitivity-table').innerHTML = tableHTML;
        }

        function switchScenario(key) {
            activeScenarioKey = key;
            ['base', 'cons', 'aggr'].forEach(k => {
                const btn = document.getElementById('sc-btn-' + k);
                if (btn) btn.style.borderColor = 'var(--border-subtle)';
            });
            const activeBtn = document.getElementById(key === 'base' ? 'sc-btn-base' : (key === 'conservative' ? 'sc-btn-cons' : 'sc-btn-aggr'));
            if (activeBtn) activeBtn.style.borderColor = 'var(--accent-cyan)';

            if (!currentVerdictData || !currentVerdictData.scenario_sensitivities) return;
            const sc = currentVerdictData.scenario_sensitivities[key];
            if (!sc) return;

            document.getElementById('wb-sc-mean').innerText = sc.mean_roi.toFixed(2) + 'x';
            document.getElementById('wb-sc-net').innerText = (sc.mean_net_return_m >= 0 ? '+$' : '-$') + Math.abs(sc.mean_net_return_m).toFixed(2) + 'M';
            document.getElementById('wb-sc-loss').innerText = (sc.prob_loss * 100).toFixed(0) + '%';
            document.getElementById('wb-sc-desc').innerText = sc.description;

            renderMonteCarloChart(key);
        }

        function renderMonteCarloChart(regime) {
            const canvas = document.getElementById('wbMonteCarloCanvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const labels = ['0x', '1x', '2x', '5x', '10x', '25x', '50x', '100x+'];
            let distribution = [];

            if (regime === 'conservative') {
                distribution = [42, 28, 18, 9, 2.5, 0.4, 0.1, 0.0];
            } else if (regime === 'aggressive') {
                distribution = [24, 18, 16, 17, 14, 7, 3, 1];
            } else {
                distribution = [32, 24, 20, 14, 7, 2.3, 0.6, 0.1];
            }

            if (monteCarloChartInstance) {
                monteCarloChartInstance.destroy();
            }

            monteCarloChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Simulated Probability (%)',
                        data: distribution,
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.15)',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            ticks: { color: '#64748b', font: { size: 9 } },
                            grid: { display: false }
                        },
                        y: {
                            ticks: { display: false },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' }
                        }
                    }
                }
            });
        }

        async function loadMarkdownPreview() {
            if (!currentVerdictData) return;
            try {
                const response = await fetch('/api/markdown_report', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentVerdictData)
                });
                const resData = await response.json();
                document.getElementById('wb-memo-markdown-text').innerText = resData.markdown;
            } catch (err) {
                document.getElementById('wb-memo-markdown-text').innerText = 'Failed to load memorandum: ' + err.message;
            }
        }

        async function copyMarkdownReport() {
            if (!currentVerdictData) return;
            try {
                const response = await fetch('/api/markdown_report', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentVerdictData)
                });
                const resData = await response.json();
                await navigator.clipboard.writeText(resData.markdown);
                alert('Investment memorandum copied to clipboard in Markdown format!');
            } catch (err) {
                alert('Failed to copy memorandum: ' + err.message);
            }
        }

        // PIPELINE PORTFOLIO ALLOCATION
        async function runPipelineAllocation() {
            const budget = parseFloat(document.getElementById('fund-budget-select').value);
            const capacity = parseInt(document.getElementById('fund-capacity-select').value);

            try {
                const response = await fetch('/api/pipeline', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        deals: PIPELINE_DEALS,
                        fund_budget_m: budget,
                        capacity_k: capacity
                    })
                });
                const resData = await response.json();
                renderPipelineResults(resData);
            } catch (err) {
                console.error("Pipeline allocation error:", err);
            }
        }

        function renderPipelineResults(data) {
            document.getElementById('pipe-capital').innerText = `$${data.total_capital_deployed_m.toFixed(2)}M / $${data.fund_budget_m.toFixed(1)}M`;
            const unallocated = data.fund_budget_m - data.total_capital_deployed_m;
            document.getElementById('pipe-unallocated').innerText = `Unallocated Reserve: $${unallocated.toFixed(2)}M`;
            document.getElementById('pipe-emv').innerText = `+$${data.total_emv_selected_m.toFixed(2)}M`;
            document.getElementById('pipe-prob').innerText = `${(data.mean_selected_probability * 100).toFixed(1)}%`;
            
            const moic = data.total_capital_deployed_m > 0 ? (data.total_emv_selected_m + data.total_capital_deployed_m) / data.total_capital_deployed_m : 0;
            document.getElementById('pipe-moic').innerText = `${moic.toFixed(1)}x MOIC`;

            const tbody = document.getElementById('pipeline-tbody');
            tbody.innerHTML = '';

            const chartLabels = [];
            const chartEMVs = [];
            const chartChecks = [];
            const chartColors = [];

            data.evaluated_deals.forEach(deal => {
                const isSelected = deal.selected;
                const rowClass = isSelected ? 'row-allocated' : 'row-rejected';
                const badgeClass = isSelected ? 'pipeline-badge-approved' : 'pipeline-badge-passed';
                const badgeText = isSelected ? 'APPROVED & ALLOCATED' : 'PASSED / BUDGET CEILING';

                tbody.innerHTML += `
                    <tr class="${rowClass}">
                        <td><span class="pipeline-badge ${badgeClass}">${badgeText}</span></td>
                        <td><strong style="color: #fff;">${deal.name}</strong></td>
                        <td>${deal.market_category.toUpperCase()} • ${deal.business_model}</td>
                        <td style="font-family: 'JetBrains Mono';">$${deal.check_size_m.toFixed(2)}M</td>
                        <td style="font-family: 'JetBrains Mono';">$${deal.target_exit_m.toFixed(1)}M</td>
                        <td style="font-family: 'JetBrains Mono'; color: var(--accent-cyan);">${(deal.probability * 100).toFixed(1)}%</td>
                        <td style="font-family: 'JetBrains Mono'; color: var(--text-muted);">${(deal.hurdle_rate * 100).toFixed(1)}%</td>
                        <td style="font-family: 'JetBrains Mono'; font-weight: 700; color: ${deal.emv_m >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">
                            ${deal.emv_m >= 0 ? '+$' : '-$'}${Math.abs(deal.emv_m).toFixed(2)}M
                        </td>
                    </tr>
                `;

                chartLabels.push(deal.name);
                chartEMVs.push(deal.emv_m);
                chartChecks.push(deal.check_size_m);
                chartColors.push(isSelected ? '#10b981' : '#64748b');
            });

            // Render Comparative Bar Chart
            const ctx = document.getElementById('pipelineBarCanvas').getContext('2d');
            if (pipelineBarChartInstance) pipelineBarChartInstance.destroy();

            pipelineBarChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: chartLabels,
                    datasets: [
                        {
                            label: 'Expected Net EMV ($M)',
                            data: chartEMVs,
                            backgroundColor: chartColors,
                            borderRadius: 6
                        },
                        {
                            label: 'Check Size ($M)',
                            data: chartChecks,
                            backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                            borderWidth: 1,
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                        y: {
                            ticks: { color: '#64748b' },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#f8fafc', font: { size: 11 } } }
                    }
                }
            });
        }

        // Auto-run evaluation on load to initialize Window 2
        window.addEventListener('DOMContentLoaded', () => {
            document.getElementById('underwriting-form').dispatchEvent(new Event('submit'));
        });
    </script>
</body>
</html>
"""


class BCAPMRequestHandler(BaseHTTPRequestHandler):
    validator = BCAPMValidator(top_k=5)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode("utf-8"))
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass
        else:
            try:
                self.send_response(404)
                self.end_headers()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if self.path == "/api/validate":
            try:
                data = json.loads(body.decode("utf-8"))
                idea = StartupIdea.from_dict(data)
                check_size = float(data.get("check_size_m", 1.0))
                target_exit = float(data.get("target_exit_m", 20.0))

                verdict = self.validator.validate_idea(
                    idea=idea,
                    check_size_m=check_size,
                    target_exit_m=target_exit,
                    capacity_k=5
                )

                response_dict = {
                    "verdict": verdict.verdict,
                    "calibrated_probability": verdict.calibrated_probability,
                    "empirical_precedent_exit_rate": verdict.empirical_precedent_exit_rate,
                    "economic_threshold": verdict.economic_threshold,
                    "expected_monetary_value_m": verdict.expected_monetary_value_m,
                    "executive_summary": verdict.executive_summary,
                    "check_size_m": check_size,
                    "target_exit_m": target_exit,
                    "idea": idea.to_dict(),
                    "retrieval_result": verdict.retrieval_result,
                    "backdrop_evaluation": verdict.backdrop_evaluation,
                    "condition_diagnostics": verdict.condition_diagnostics,
                    "scenario_sensitivities": verdict.scenario_sensitivities
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_dict).encode("utf-8"))

            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass
            except Exception as e:
                logger.error(f"Error handling validation request: {e}")
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    pass

        elif self.path == "/api/markdown_report":
            try:
                data = json.loads(body.decode("utf-8"))
                idea_dict = data.get("idea", {})
                idea = StartupIdea.from_dict(idea_dict)
                check_size = float(data.get("check_size_m", 1.0))
                target_exit = float(data.get("target_exit_m", 20.0))

                verdict = self.validator.validate_idea(
                    idea=idea,
                    check_size_m=check_size,
                    target_exit_m=target_exit,
                    capacity_k=5
                )
                md_str = format_verdict_report(verdict, as_markdown=True)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"markdown": md_str}).encode("utf-8"))
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass
            except Exception as e:
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    pass

        elif self.path == "/api/pipeline":
            try:
                data = json.loads(body.decode("utf-8"))
                deals = data.get("deals", [])
                fund_budget_m = float(data.get("fund_budget_m", 5.0))
                capacity_k = int(data.get("capacity_k", 2))

                evaluated_deals = []
                probabilities = []
                emvs = []
                check_sizes = []

                for d in deals:
                    idea = StartupIdea(
                        name=d.get("name", "Venture"),
                        market_category=d.get("market_category", "software"),
                        business_model=d.get("business_model", "B2B"),
                        country_code=d.get("country_code", "USA"),
                        city=d.get("city", ""),
                        initial_funding_usd=float(d.get("check_size_m", 1.0)) * 1_000_000.0,
                        founder_count=int(d.get("founder_count", 2)),
                        worked_in_top_companies=bool(d.get("worked_in_top_companies", False)),
                        cax_cofounders=bool(d.get("cax_cofounders", False)),
                        repeat_investor_count=int(d.get("repeat_investor_count", 0)),
                        is_ml_based=bool(d.get("is_ml_based", False))
                    )
                    c_m = float(d.get("check_size_m", 1.0))
                    v_m = float(d.get("target_exit_m", 20.0))

                    verdict = self.validator.validate_idea(idea=idea, check_size_m=c_m, target_exit_m=v_m)
                    prob = verdict.calibrated_probability
                    emv = verdict.expected_monetary_value_m
                    hurdle = verdict.economic_threshold

                    probabilities.append(prob)
                    emvs.append(emv)
                    check_sizes.append(c_m)

                    evaluated_deals.append({
                        "name": d.get("name"),
                        "market_category": d.get("market_category"),
                        "business_model": d.get("business_model"),
                        "check_size_m": c_m,
                        "target_exit_m": v_m,
                        "probability": round(prob, 4),
                        "hurdle_rate": round(hurdle, 4),
                        "emv_m": round(emv, 2),
                        "selected": False
                    })

                # Greedy capacity-constrained allocation by highest EMV within capital budget and capacity K
                ranked_order = np.argsort(emvs)[::-1]
                deployed_capital = 0.0
                selected_count = 0
                selected_emv = 0.0

                for idx in ranked_order:
                    deal_c = check_sizes[idx]
                    if selected_count < capacity_k and (deployed_capital + deal_c) <= fund_budget_m and emvs[idx] > 0:
                        evaluated_deals[idx]["selected"] = True
                        deployed_capital += deal_c
                        selected_count += 1
                        selected_emv += emvs[idx]

                selected_probs = [evaluated_deals[i]["probability"] for i in range(len(evaluated_deals)) if evaluated_deals[i]["selected"]]
                mean_sel_p = float(np.mean(selected_probs)) if selected_probs else 0.0

                resp = {
                    "fund_budget_m": fund_budget_m,
                    "capacity_k": capacity_k,
                    "total_capital_deployed_m": round(deployed_capital, 2),
                    "total_emv_selected_m": round(selected_emv, 2),
                    "mean_selected_probability": round(mean_sel_p, 4),
                    "evaluated_deals": evaluated_deals
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))

            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass
            except Exception as e:
                logger.error(f"Error handling pipeline request: {e}")
                try:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    pass
        else:
            try:
                self.send_response(404)
                self.end_headers()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass


def run_server(port: int = 8080):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, BCAPMRequestHandler)
    print("=" * 75)
    print(f"  BCAPM INSTITUTIONAL VENTURE UNDERWRITING TERMINAL ACTIVE")
    print(f"  Access Clean Windowed Architecture at: http://localhost:{port}")
    print("=" * 75)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    import os
    default_port = int(os.environ.get("PORT", 8080))
    parser = argparse.ArgumentParser(description="Run BCAPM interactive web validator server.")
    parser.add_argument("--port", type=int, default=default_port, help=f"HTTP port to serve on (default: {default_port})")
    args = parser.parse_args()
    run_server(port=args.port)
