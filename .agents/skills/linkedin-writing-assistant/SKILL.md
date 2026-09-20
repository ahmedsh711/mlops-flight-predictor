---
name: linkedin-writing-assistant
description: Turn a technical update — finishing a course session, shipping a project, passing a certification, a work milestone — into a natural, human-sounding LinkedIn post in Egyptian Arabic with English technical terms kept as-is. Always use this skill when the user asks for a LinkedIn post, asks to "document" or "share" a project/learning journey, or asks to make a draft sound more human/natural/less AI-polished — even if they don't say "LinkedIn" explicitly (e.g. "اكتبلي بوست", "عاوز اوثق الرحلة", "خليه يبان بشري", "ظبط الدرافت ده"). Also use when reviewing or fixing an existing draft post so it stops reading as AI-generated.
---

# LinkedIn Writing Assistant

Writes LinkedIn posts about technical work that read like a real engineer wrote them at 11pm, not like a content template. The rules below matter more than covering every detail of what the user learned or built — a shorter, sharper, honest post beats a comprehensive AI-polished one every time.

## Before writing: gather what you need

Pull the concrete material from what the user gave you (slides, transcripts, code, READMEs, a certificate, an earlier draft) before writing a single line. The post lives or dies on specifics: real service/tool names, a real number, a real decision and why, a real thing that went wrong. Never invent a personal experience, a struggle, or a timeframe the user hasn't given you — ground everything in what's actually there, or leave it out.

If the user hands you a rough draft, keep their actual details and their voice — polish structure and flow, don't replace their content with generic phrasing.

## Language

Egyptian Arabic, colloquial — not stiff Modern Standard Arabic, not a formal announcement register. Keep technical terms, tool names, and product names in English exactly as the industry writes them (Docker, FastAPI, ONNX, pytest, AWS, EC2, MLflow…). Never write full English sentences unless the user explicitly asks for an English version alongside the Arabic one.

## The hook (first 1–2 lines)

Open with something small, concrete, and human — never a crafted marketing question. What works:
- A specific, real detail from the work itself (a funny filename, an expectation that turned out wrong, a mundane moment, a plain "before" state).
- A short, honest statement with no adjectives doing the work for you.

What to avoid: rhetorical questions engineered to "stop the scroll," a grand opening statement, anything vague enough that it could open a hundred other people's posts about the same topic.

## Structure — pick one, don't mix

**Flowing narrative** — default for anything telling a story: a project, a realization, an experience, a "here's what happened" post. Each paragraph leads into the next with a real transition ("بعدين واجهت...", "وده جابني لـ...", "المشكلة إن...") — never a list of disconnected features.

**Numbered-emoji list (1️⃣2️⃣3️⃣…)** — only when there are several genuinely separate, parallel points worth scanning (e.g. a session recap covering 5–7 distinct topics). Never assign a different themed emoji to each point (📦⚙️🔄🐳📝✅🎯…) — that specific pattern reads as AI-generated faster than almost anything else.

Either way: short paragraphs, real line breaks, no wall of text. No markdown headers or `**bold**` syntax inside the post body — LinkedIn doesn't render them, so they'd show up as literal asterisks.

## Voice — the anti-AI checklist

**Never use these phrases, in Arabic or English, under any framing:** "thrilled to announce" / "excited to share my journey" / "honored and humbled" / "grateful beyond words" / "blessed and grateful" / "this is just the beginning" / "couldn't be more excited" / "so proud to share" — and their Arabic equivalents (زي "سعيد جدًا وفخور إني أشارك رحلتي").

**Avoid:**
- Perfectly parallel sentence structure repeated across every paragraph
- A philosophical or "lesson learned" opening statement
- A neatly resolved, bow-tied ending where every thread is wrapped up
- Adjective-stacking ("incredible," "powerful," "amazing," "رهيب," "مبهر")
- Smooth connector phrases that string every paragraph together identically ("وهنا كانت النقطة اللي..." repeated three times)

**Do:**
- Vary sentence length on purpose — a few short lines, then one longer one
- Let the post end a little unresolved rather than fully wrapped up
- Name the actual hardest or most annoying part of the work instead of listing tools — specificity about a real struggle reads as more human than any adjective ever could
- Allow small, natural imperfection — this is a person talking, not a press release

## Content

A tool-listicle ("استخدمت Docker وFastAPI وMLflow!") reads as AI-generated even when every word is true. What makes a post read as human is the *why*: why this tool and not the obvious alternative, what broke, what took longer than expected, what surprised the person doing the work. Prefer one well-told specific decision over five tools name-dropped in a row.

## Emojis

Minimal, and only when they carry real information — numbering a sequence, or a genuine one-off reaction (a funny coincidence, a real "😅" moment). Never one emoji per section as decoration, never emoji bullets on every line.

## Links

Never put a raw link inside the post body — it measurably hurts reach on LinkedIn. Provide it separately, clearly labeled as a suggested first comment, at the end of the file.

## Output

Deliver the finished post as a file (not just inline chat text) — it's something the user will copy-paste elsewhere, so treat it as a standalone artifact:
- Save it as a `.md` file (plain paragraphs, no literal markdown syntax the user would have to strip out)
- Present it to the user rather than pasting the full text back into the chat reply
- Keep the chat reply itself short: one or two lines on what you changed or why, no restating the post

## Quick pre-delivery checklist

- [ ] No banned phrase, in either language
- [ ] Hook is one specific concrete detail, not a rhetorical question
- [ ] Structure is either flowing narrative OR numbered list — not a mix
- [ ] Sentence lengths actually vary
- [ ] At least one real specific (a number, a name, a decision, a struggle) — not just tool names
- [ ] Links moved out of the body into a suggested-comment block
- [ ] Emojis are functional, not decorative
- [ ] Saved and presented as a file, chat reply stays short