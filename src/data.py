"""Dataset loading and generation for Persian support-ticket intents."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATASET_FILE, LABEL_COLUMN, LABELS, TEXT_COLUMN

# Curated templates — realistic support-desk phrasing (not scraped copyrighted text)
_TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "مبلغ صورتحساب این ماه بیشتر از حد انتظار است لطفا بررسی کنید",
        "پرداخت آنلاین انجام دادم اما هنوز وضعیت بدهی فعال است",
        "کد تخفیف روی فاکتور اعمال نشده است",
        "میخوام جزئیات هزینه ترافیک اضافه را ببینم",
        "رسید پرداخت کارت به کارت را فرستادم لطفا تأیید کنید",
        "چرا دو بار از حسابم برای یک سرویس کسر شده؟",
        "لطفا فاکتور رسمی با کد اقتصادی برای شرکت ارسال کنید",
        "تاریخ سررسید پرداخت را اشتباه ثبت کرده‌اید",
        "می‌خواهم روش پرداخت را از کارت به پایا تغییر دهم",
        "هزینه نصب اولیه روی صورتحساب تکرار شده است",
        "اعتبار کیف پول من کم شده بدون دلیل مشخص",
        "لطفا بدهی معوق را به صورت قسطی تسویه کنیم",
    ],
    "technical": [
        "اینترنت از صبح قطع است و چراغ PON چشمک می‌زند",
        "سرعت دانلود خیلی پایین‌تر از پلن من است",
        "قطع و وصل مکرر روی وای‌فای داخلی دارم",
        "پینگ بازی بالا رفته و لگ دارم",
        "مودم بعد از ریستارت به شبکه وصل نمی‌شود",
        "از دیشب لینک فیبر قطع شده لطفا تیکت بزنید",
        "در برخی سایت‌ها تایم‌اوت می‌گیرم",
        "آپلود فایل روی سرور شرکتی شکست می‌خورد",
        "سیگنال موبایل روی آنتن شما ضعیف است",
        "بعد از قطعی برق کانفیگ مودم از بین رفت",
        "پورت لن مودم کار نمی‌کند",
        "DNS به درستی resolve نمی‌کند",
    ],
    "sales": [
        "می‌خواهم پلن اینترنت را به ۵۰ مگ ارتقا دهم",
        "آیا پلن سازمانی برای شرکت ده نفره دارید؟",
        "هزینه راه‌اندازی فیبر برای آدرس جدید چقدر است؟",
        "لطفا شرایط خرید مودم اقساطی را بگویید",
        "برای جابجایی سرویس به آدرس دیگر چه مدارکی لازم است؟",
        "پکیج ویژه نوروز هنوز فعال است؟",
        "می‌خواهم خط تلفن ثابت را همراه اینترنت بگیرم",
        "تفاوت پلن نامحدود و حجمی چیست؟",
        "آیا پوشش فیبر در محله ما برقرار است؟",
        "درخواست مشاوره برای انتخاب پلن مناسب دارم",
        "می‌خواهم سرویس دوم برای مغازه ثبت کنم",
        "کد معرف برای ثبت‌نام جدید دارم",
    ],
    "cancellation": [
        "لطفا سرویس اینترنت را در پایان دوره لغو کنید",
        "می‌خواهم اشتراک را برای همیشه قطع کنم",
        "به دلیل اسباب‌کشی دیگر به سرویس نیاز ندارم",
        "فرم فسخ قرارداد را چطور دریافت کنم؟",
        "لغو را ثبت کردم اما هنوز از حسابم کم می‌شود",
        "می‌خواهم سرویس را موقتا به حالت تعلیق ببرم",
        "لطفا تجهیزات امانی را برای جمع‌آوری هماهنگ کنید",
        "به اپراتور دیگری مهاجرت می‌کنم درخواست پورت‌اوت دارم",
        "انصراف از طرح تعهد دوساله",
        "حساب کاربری را حذف کنید",
        "سرویس را تا اطلاع ثانوی غیرفعال کنید",
        "درخواست لغو به دلیل کیفیت نامناسب",
    ],
    "other": [
        "ساعت کار پشتیبانی حضوری را می‌خواستم",
        "آدرس دفتر مرکزی را لطفا ارسال کنید",
        "می‌خواهم از پرسنل نصب تشکر کنم",
        "پیشنهاد بهبود اپلیکیشن موبایل شما را دارم",
        "خبرنامه ایمیل برایم زیاد ارسال می‌شود",
        "لطفا راهنمای تنظیمات اولیه را بفرستید",
        "سوال عمومی درباره پوشش شهرهای دیگر",
        "می‌خواهم با مدیر منطقه صحبت کنم",
        "اطلاعات تماس پشتیبانی شبانه",
        "درخواست معرفی نماینده فروش محلی",
        "سوال درباره فرصت شغلی در شرکت شما",
        "پیام آزمایشی برای بررسی سامانه تیکت",
    ],
}


def _expand_templates() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    suffixes = [
        "",
        " ممنون",
        " لطفا پیگیری کنید",
        " فوریت دارد",
        " با احترام",
        " شماره پرونده قبلی را هم داشتم",
    ]
    for label, texts in _TEMPLATES.items():
        for t in texts:
            for s in suffixes:
                rows.append({TEXT_COLUMN: (t + s).strip(), LABEL_COLUMN: label})
    return pd.DataFrame(rows)


def build_dataset(force: bool = False) -> pd.DataFrame:
    """Create or load the local CSV dataset."""
    if DATASET_FILE.exists() and not force:
        df = pd.read_csv(DATASET_FILE)
    else:
        df = _expand_templates()
        DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_FILE, index=False)

    assert TEXT_COLUMN in df.columns and LABEL_COLUMN in df.columns
    unknown = set(df[LABEL_COLUMN].unique()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels in dataset: {unknown}")
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)
    return df


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATASET_FILE
    if not path.exists():
        return build_dataset()
    return pd.read_csv(path)
