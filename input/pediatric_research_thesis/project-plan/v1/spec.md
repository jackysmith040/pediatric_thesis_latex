# Pediatric Patient Counting V1

## Overview

A dual-stack system connecting a Python FastAPI YOLOv8 engine to a Laravel 13 Dashboard. It counts adults and children in a waiting room and displays it via a real-time glassmorphism Livewire UI.

## Goals

1. Detect adults and children in real-time.
2. Send data to Laravel Restify and broadcast to UI via Reverb.
3. Automatically generate daily capacity PDFs.

## Core User Flow

1. Admin opens Dashboard.
2. CV Engine processes RTSP/webcam feed.
3. Dashboard updates automatically.

## Scope

### In Scope
- FastAPI Engine updates
- Dashboard telemetry ingestion
- Basic analytics and night PDF generation

### Out of Scope
- User authentication
- Complex historical data visualization
