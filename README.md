# GENIUS BOT MAN — Execution Worker

This repository contains ONLY the Execution Layer.

## Rules (Non-Negotiable)
- This bot NEVER makes decisions
- This bot NEVER generates signals
- This bot executes ONLY certified signals from SIGNAL_OUTBOX
- Risk rules, limits, and kill-switch are mandatory
- DEMO → LIVE transition is governed by DB flags & human approval

## What lives here
- Execution engine (Python)
- Exchange adapter (ccxt / Binance)
- Virtual Wallet (DEMO)
- Startup Sync & Risk Guards

## What does NOT live here
- Strategy logic
- Decision logic
- Excel brain
- Signal generation

## Developer Instructions
Read `/specs/developer_job_spec.md` before writing any code.

Any deviation from specs is considered a violation.
