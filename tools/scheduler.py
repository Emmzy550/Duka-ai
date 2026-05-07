"""Report scheduling: save preferences, send emails, run background checks."""
from __future__ import annotations

import json
import os
import smtplib
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any


SCHEDULES_FILE = Path("user_schedules.json")


# ── Persistence ───────────────────────────────────────────────────────────────

def load_schedules() -> dict[str, Any]:
    if SCHEDULES_FILE.exists():
        try:
            return json.loads(SCHEDULES_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_schedules(schedules: dict[str, Any]) -> None:
    SCHEDULES_FILE.write_text(json.dumps(schedules, indent=2), encoding="utf-8")


# ── Schedule timing ───────────────────────────────────────────────────────────

def calculate_next_send(frequency: str) -> str:
    now = datetime.now()
    if frequency == "weekly":
        days_ahead = 7 - now.weekday()
        if days_ahead == 0:
            days_ahead = 7
        target = now + timedelta(days=days_ahead)
    elif frequency == "monthly":
        if now.month == 12:
            target = now.replace(year=now.year + 1, month=1, day=1)
        else:
            target = now.replace(month=now.month + 1, day=1)
    elif frequency == "biweekly":
        target = now + timedelta(days=14)
    elif frequency == "quarterly":
        target = now + timedelta(days=90)
    else:
        target = now + timedelta(days=7)
    return target.replace(hour=8, minute=0, second=0, microsecond=0).isoformat()


def save_schedule(email: str, frequency: str, business_context: dict[str, Any]) -> str:
    schedules = load_schedules()
    schedule_id = f"user_{len(schedules) + 1}"
    schedules[schedule_id] = {
        "email": email,
        "frequency": frequency,
        "business_context": business_context,
        "created_at": datetime.now().isoformat(),
        "last_sent": None,
        "next_send": calculate_next_send(frequency),
        "active": True,
    }
    save_schedules(schedules)
    return schedule_id


def delete_schedule(schedule_id: str) -> None:
    schedules = load_schedules()
    if schedule_id in schedules:
        schedules[schedule_id]["active"] = False
        save_schedules(schedules)


# ── Email sending ─────────────────────────────────────────────────────────────

def send_report_email(
    recipient_email: str,
    business_name: str,
    report_text: str,
    frequency: str = "weekly",
    pdf_path: str | None = None,
) -> bool:
    sender_email = os.getenv("GMAIL_ADDRESS", "").strip()
    sender_password = os.getenv("GMAIL_APP_PASSWORD", "").strip()

    if not sender_email or not sender_password:
        raise EnvironmentError("GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in .env")

    freq_label = {
        "weekly": "Weekly",
        "monthly": "Monthly",
        "biweekly": "Bi-Weekly",
        "quarterly": "Quarterly",
    }.get(frequency, "Scheduled")

    msg = MIMEMultipart()
    msg["From"] = f"Duka AI <{sender_email}>"
    msg["To"] = recipient_email
    msg["Subject"] = f"Your {freq_label} Financial Report — {business_name}"

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;
                 background: #f5f5f5; padding: 0;">
        <div style="background: linear-gradient(135deg, #1A3A5C 0%, #0F172A 100%);
                    padding: 28px 24px; text-align: center;">
            <h1 style="color: #FFFFFF; margin: 0; font-size: 22px; letter-spacing: 0.05em;">
                Duka AI
            </h1>
            <p style="color: #1D9E75; margin: 6px 0 0; font-size: 13px; font-weight: 600;">
                Your {freq_label} Financial Report
            </p>
        </div>

        <div style="padding: 28px 24px; background: #f9f9f9;">
            <h2 style="color: #1A3A5C; margin-top: 0;">Hello!</h2>
            <p style="color: #444; font-size: 14px;">
                Here is your automated financial report for
                <strong>{business_name}</strong>.
            </p>
            <div style="background: #FFFFFF; padding: 20px; border-radius: 8px;
                        border-left: 4px solid #1D9E75;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.08);">
                <pre style="white-space: pre-wrap; font-family: monospace; font-size: 13px;
                            color: #222; margin: 0; line-height: 1.6;">{report_text}</pre>
            </div>
        </div>

        <div style="background: #0A1628; padding: 16px 24px; text-align: center;">
            <p style="color: #666; font-size: 11px; margin: 0; line-height: 1.8;">
                Sent automatically by Duka AI &nbsp;&middot;&nbsp;
                Powered by AMD MI300X &nbsp;&middot;&nbsp; Qwen Model
            </p>
        </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(html_body, "html"))

    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            attachment = MIMEBase("application", "octet-stream")
            attachment.set_payload(f.read())
            encoders.encode_base64(attachment)
            attachment.add_header(
                "Content-Disposition",
                "attachment; filename=DukaAI_Report.pdf",
            )
            msg.attach(attachment)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

    return True


# ── Automated report generation ───────────────────────────────────────────────

def _build_report_text(business_context: dict[str, Any]) -> str:
    """Generate a plain-text report from saved business context."""
    from agents.orchestrator import generate_business_report
    from agents.report_agent import generate_full_report
    from tools.financial_engine import compute_metrics, get_baseline

    try:
        report = generate_business_report(
            business_context.get("manual_notes", "Automated scheduled report."),
            business_profile={
                "business_type": business_context.get("business_type", ""),
                "location": business_context.get("location", "Zambia"),
                "products_services": business_context.get("products_services", ""),
            },
        )
        baseline = get_baseline(business_context) or {}
        metrics = compute_metrics(baseline) if baseline else {}
        full = generate_full_report(baseline, metrics, report)

        snap = full.get("snapshot", {})
        loan_d = full.get("loan", {})
        recs = full.get("recommendations", [])
        risks = full.get("risks", [])

        recs_text = "\n".join(
            f"{i + 1}. {r.get('action', '')} -> {r.get('impact', '')} ({r.get('timeline', '')})"
            for i, r in enumerate(recs)
        )
        risks_text = (
            "RISKS TO WATCH\n" + "\n".join(f"  {r}" for r in risks) + "\n\n"
            if risks else ""
        )

        return (
            f"Duka AI -- Financial Health Report\n"
            f"{full.get('business_name', 'Your Business')} | {full.get('location', 'Zambia')}"
            f" | Generated {full.get('generated_at', '')}\n"
            f"{'=' * 60}\n\n"
            f"EXECUTIVE SUMMARY\n{full.get('executive_summary', '')}\n\n"
            f"FINANCIAL SNAPSHOT\n"
            f"Revenue:      K{snap.get('revenue', 0):,.0f}\n"
            f"Expenses:     K{snap.get('expenses', 0):,.0f} ({snap.get('expense_pct', 0):.0f}%)\n"
            f"Profit:       K{snap.get('profit', 0):,.0f} ({snap.get('margin_pct', 0):.1f}% margin)\n"
            f"Health Score: {snap.get('health_score', 0)}/100 -- {snap.get('health_label', '')}\n\n"
            f"LOAN READINESS\n"
            f"Score:          {loan_d.get('score', 0)}/100 -- {loan_d.get('status', '')}\n"
            f"Safe borrowing: K{loan_d.get('safe_amount', 0):,.0f}\n\n"
            f"TOP RECOMMENDATIONS\n{recs_text}\n\n"
            f"{risks_text}"
            f"{'=' * 60}\n"
            f"Generated by Duka AI | Powered by AMD MI300X | Qwen Model"
        )
    except Exception as exc:
        return (
            f"Duka AI -- Automated Report\n\n"
            f"Report generation encountered an issue: {exc}\n"
            f"Please log in to the Duka AI app to generate your report manually."
        )


# ── Background scheduler ──────────────────────────────────────────────────────

def check_and_send_reports() -> None:
    """Check all active schedules and send any that are due. Called hourly."""
    schedules = load_schedules()
    now = datetime.now()
    changed = False

    for schedule_id, schedule in schedules.items():
        if not schedule.get("active"):
            continue
        try:
            next_send = datetime.fromisoformat(schedule["next_send"])
        except (KeyError, ValueError):
            continue

        if now >= next_send:
            try:
                ctx = schedule.get("business_context", {})
                report_text = _build_report_text(ctx)
                send_report_email(
                    recipient_email=schedule["email"],
                    business_name=ctx.get("business_type") or "Your Business",
                    report_text=report_text,
                    frequency=schedule.get("frequency", "weekly"),
                )
                schedule["last_sent"] = now.isoformat()
                schedule["next_send"] = calculate_next_send(schedule.get("frequency", "weekly"))
                changed = True
            except Exception:
                pass  # keep running even if one send fails

    if changed:
        save_schedules(schedules)


def start_scheduler() -> None:
    """Start the APScheduler background thread — idempotent via st.cache_resource."""
    import streamlit as st

    @st.cache_resource
    def _cached_start():
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            scheduler = BackgroundScheduler(daemon=True)
            scheduler.add_job(check_and_send_reports, "interval", hours=1)
            scheduler.start()
            return scheduler
        except Exception:
            return None

    _cached_start()
