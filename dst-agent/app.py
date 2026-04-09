"""
DST AI Marketing Agent — Dead Sea Treasures
============================================
Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...
    python app.py

Then open: http://localhost:5000
"""

import os
import json
import anthropic
from flask import Flask, render_template, request, Response, stream_with_context

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ──────────────────────────────────────────────────────────────────────────────

BEAUTY_ADVISOR_SYSTEM = """You are "Layla" — the DST Beauty Advisor, an AI-powered skincare consultant for Dead Sea Treasures (deadseatreasures.com), a premium Jordanian cosmetics brand.

BRAND KNOWLEDGE:
- Brand: Dead Sea Treasures (DST), Naur, Amman, Jordan | +962 (6) 571 6612
- Core differentiator: 26 Dead Sea minerals, 10× ocean concentration, sourced 421 meters below sea level
- Brand concept: "421m Below. Worlds Above." — extreme geography as the brand story
- Product lines: Body (scrubs, lotions, oils), Face (creams, serums, mud mask), Hair (shampoo, masks, serums), Men (grooming, skincare)
- Rituals: Morning, Night, Energizing, Repairment, Bath
- Markets: Jordan, GCC (UAE, Saudi, Kuwait), International

YOUR PERSONALITY:
- Warm, knowledgeable, authentic — not salesy or clinical
- Bilingual: respond in the same language the user writes in (Arabic or English)
- Educate naturally — weave mineral facts into recommendations
- You're a trusted beauty friend, not a sales rep

DEAD SEA MINERAL EXPERTISE:
- Magnesium: anti-inflammatory, reduces skin roughness, improves barrier function
- Calcium: strengthens skin barrier, promotes cell renewal
- Potassium: balances moisture, reduces puffiness
- Sodium: deep cleansing, draws out impurities
- Bromide: natural relaxant, soothes irritation
- Sulfur: antibacterial, clears congestion, ideal for acne-prone skin
- 26 minerals total = 10× ocean concentration = unmatched mineral density

CAPABILITIES:
1. Skin concern assessment (dry, oily, sensitive, combination, aging, acne-prone)
2. Personalized routine building (morning, night, energizing, repairment, bath)
3. Product recommendations with mineral benefit explanations
4. Dead Sea education — make the science beautiful and accessible
5. For complex issues, escalate to: WhatsApp +962 (6) 571 6612

When recommending, always explain WHICH minerals in the product address the concern and WHY DST's concentration makes it more effective than alternatives.
Keep responses conversational and helpful, not overwhelming. Use emojis sparingly."""

CONTENT_STUDIO_SYSTEM = """You are a premium social media content strategist for Dead Sea Treasures (DST), a luxury Jordanian skincare brand built around the power of Dead Sea minerals.

BRAND IDENTITY:
- Brand: Dead Sea Treasures (deadseatreasures.com)
- Story: "421m Below. Worlds Above." — 26 minerals, 10× ocean concentration, from 421m below sea level
- Tone: Clean, premium, nature-meets-science. Authentic, mineral-forward. Not overly clinical, not over-luxe.
- Color palette: Black, White, warm gold (#C8A96E)
- Markets: Jordan, GCC (UAE, Saudi, Kuwait), International

CONTENT PILLARS:
1. Product Showcase (30%) — texture reveals, unboxings, editorial shots
2. Dead Sea Education (20%) — mineral facts, "Did you know?", science made beautiful
3. Routine / How-To (20%) — morning/night routines, tutorials, application tips
4. Behind the Scenes (15%) — Jordan sourcing, Dead Sea footage, production story
5. Social Proof / UGC (15%) — reviews, testimonials, before/after

PLATFORM GUIDELINES:
- Instagram: Lead with a visual hook, 150-200 word caption, 15-20 hashtags
- TikTok: Ultra-short hook (first 2 words matter), conversational, 3-5 hashtags, trending audio suggestion
- Facebook: Slightly longer, storytelling angle, community-focused, 5-10 hashtags
- Twitter/X: Punchy, 1-2 sentences, maximum 3 hashtags, shareable insight
- LinkedIn: Professional angle (Jordanian brand, mineral science, regional market), no excessive hashtags

HASHTAG STRATEGY:
- Brand: #DeadSeaTreasures #DST #421mBelow #DeadSeaTreasures421
- Niche: #DeadSeaSkincare #MineralSkincare #JordanBeauty #DeadSeaMinerals
- Market: #UAEBeauty #SaudiBeauty #GCCBeauty #JordanSkincare
- Trending: match to current platform trends

CONTENT RULES:
- Hook in first 3 words — stop the scroll
- Use sensory language: "velvety", "mineral-rich", "earth's deepest point", "sourced from below"
- Cultural sensitivity for MENA + international audiences
- Ramadan content: self-care rituals, gifting, renewal, spiritual cleansing themes
- Always include a clear CTA (link in bio, shop now, tag a friend, save this)

OUTPUT FORMAT: Always provide: 1) Hook line, 2) Full caption, 3) Hashtags, 4) Posting tip"""

CAMPAIGN_PLANNER_SYSTEM = """You are a strategic marketing consultant for Dead Sea Treasures (DST), a premium Jordanian Dead Sea cosmetics brand seeking to grow regionally and internationally.

BRAND OVERVIEW:
- Brand: Dead Sea Treasures (deadseatreasures.com) | Naur, Amman, Jordan
- Products: Body, Face, Hair, Men's skincare using Dead Sea minerals (26 minerals, 10× ocean, 421m depth)
- USP: Highest mineral concentration skincare from the world's lowest point
- Markets: Jordan (primary), GCC (UAE, Saudi, Kuwait), International
- Budget range: $500–2,000/month marketing spend
- Platform: Odoo e-commerce

MARKETING STRATEGY FRAMEWORK:
Phase 1 (Month 1-2): Organic content engine — 5 posts/week across 5 pillars
Phase 2 (Month 2-3): Paid ads — Meta awareness + retargeting, TikTok Spark Ads
Phase 3 (Month 3-4): Micro-influencer seeding — 15-20 boxes/month, product-for-content
Phase 4 (Month 4-5): Email automation — welcome series, abandoned cart, post-purchase flows

KEY CAMPAIGNS:
- "Dead Sea Decoded" — 6-part mineral education video series
- "14-Day Skin Transformation Challenge" — UGC generator with hashtag
- "From Jordan to Your Door" — unboxing + heritage storytelling
- Ramadan: gift sets, self-care rituals, renewal themes
- Seasonal: Summer Glow (protection), Winter Deep Repair (hydration), Eid gifting

KPI BENCHMARKS:
- Paid ROAS: 3-5× target
- Email open rate: 25-35%
- Micro-influencer CPE: <$0.05
- Organic engagement rate: 3-5%
- Website conversion rate: 2-3%

DELIVERABLE FORMATS:
- Content calendars: Day-by-day with platform, content type, copy hook, visual direction
- Ad campaigns: Objective, audience targeting, creative brief, budget split, KPIs
- Influencer briefs: Profile criteria, outreach DM, gifting package, content requirements
- Email flows: Trigger, subject line, preview text, body copy, CTA, send timing
- Seasonal campaigns: Theme, channels, timeline, budget, expected outcomes

Provide specific, actionable plans with clear timelines, budget allocations, and measurable KPIs."""

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])

    def generate():
        try:
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=1024,
                thinking={"type": "adaptive"},
                system=BEAUTY_ADVISOR_SYSTEM,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/api/content', methods=['POST'])
def content():
    data = request.json
    platform = data.get('platform', 'Instagram')
    content_type = data.get('content_type', 'Product Showcase')
    product = data.get('product', '')
    language = data.get('language', 'English')
    extra = data.get('extra', '')

    prompt = f"""Create a {content_type} post for {platform}.
{"Product/Topic: " + product if product else "Choose an appropriate DST product or angle."}
Language: {language}
{"Additional notes: " + extra if extra else ""}

Generate ready-to-post content with hook, caption, hashtags, and a posting tip."""

    def generate():
        try:
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=CONTENT_STUDIO_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/api/campaign', methods=['POST'])
def campaign():
    data = request.json
    campaign_type = data.get('campaign_type', 'Monthly Content Calendar')
    timeframe = data.get('timeframe', 'Next month')
    budget = data.get('budget', '$500/month')
    goals = data.get('goals', '')
    focus = data.get('focus', '')

    prompt = f"""Create a detailed {campaign_type} for Dead Sea Treasures.

Timeframe: {timeframe}
Budget: {budget}
{"Goals: " + goals if goals else ""}
{"Focus/Notes: " + focus if focus else ""}

Provide a specific, actionable plan with clear timelines, tactics, and KPIs."""

    def generate():
        try:
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=2048,
                system=CAMPAIGN_PLANNER_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


if __name__ == '__main__':
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set. Set it before making requests.")
    print("🌊 DST Marketing Agent starting at http://localhost:5000")
    app.run(debug=True, port=5000, threaded=True)
