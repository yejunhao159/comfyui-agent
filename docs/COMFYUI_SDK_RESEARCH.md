# ComfyUI SDK & API Research Report

## Table of Contents
1. [@saintno/comfyui-sdk - Deep Dive](#1-saintnocomfyui-sdk)
2. [@stable-canvas/comfyui-client - Alternative](#2-stable-canvascomfyui-client)
3. [Head-to-Head Comparison](#3-head-to-head-comparison)
4. [Python ComfyUI Client Libraries](#4-python-comfyui-client-libraries)
5. [ComfyUI Server API Surface (Complete Reference)](#5-comfyui-server-api-surface)

---

## 1. @saintno/comfyui-sdk

**Repository:** https://github.com/comfy-addons/comfyui-sdk
**NPM:** https://www.npmjs.com/package/@saintno/comfyui-sdk
**License:** MIT | **Stars:** 241 | **Forks:** 26 | **Open Issues:** 7

### What It Does

A TypeScript-first SDK for interacting with ComfyUI's backend API. It provides:
- A fluent **PromptBuilder** pattern for mapping workflow JSON to named inputs/outputs
- A **CallWrapper** for executing workflows with lifecycle event callbacks
- A **ComfyPool** for distributing jobs across multiple ComfyUI instances
- Real-time WebSocket event streaming (progress, previews, errors)
- Authentication support (Basic Auth, Bearer Token, Custom Headers)

### Connection Mechanism

Uses **both HTTP and WebSocket**:
- **HTTP** for submitting prompts, querying queue/history, uploading images, fetching system stats
- **WebSocket** at `ws(s)://{host}/ws?clientId={uuid}` for real-time execution events
- Automatic reconnection with exponential backoff (base delay * 2^attempt, max 15s, +/-30% jitter, max 10 attempts)
- Polling fallback when WebSocket is unavailable (2-second interval)

### NPM Stats

- **Latest Version:** 0.2.49
- **Total Published Versions:** 80+
- **Weekly Downloads:** ~23,000 (week of Jan 31 - Feb 6, 2026)
- **Runtime Dependencies:** Only `ws` (^8.18.0)
- **Peer Dependencies:** TypeScript ^5.0.0
- **Package Size:** ~180-313KB unpacked
- **Maintainer:** saintno (tctien342)
- **Last Commit:** December 24, 2024

### Maturity Assessment

- Active development through 2024, with 80+ versions published
- Single maintainer (tctien342/saintno)
- 7 open issues, 1 open PR
- Good TypeScript types with full generics support
- Includes type definitions for samplers (28 types) and schedulers (6 types)
- Extension support for ComfyUI Manager and Crystools
- Last commit was December 2024 -- development may have slowed

### Source Code Architecture

```
src/
  client.ts          -- ComfyApi class (main HTTP + WS client)
  socket.ts          -- WebSocketClient (connection management)
  prompt-builder.ts  -- PromptBuilder (workflow configuration)
  call-wrapper.ts    -- CallWrapper (execution lifecycle)
  pool.ts            -- ComfyPool (multi-instance management)
  tools.ts           -- Utility functions (seed(), etc.)
  contansts.ts       -- Constants
  types/
    api.ts           -- API response/request types
    event.ts         -- WebSocket event types
    sampler.ts       -- Sampler/scheduler name types
    error.ts         -- Custom error types
    manager.ts       -- ComfyUI Manager types
    tool.ts          -- Utility types
  features/          -- Extension implementations
```
