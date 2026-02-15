import pandas as pd
from datetime import datetime

def unify_timezone(dt: datetime, index: pd.Index) -> datetime:
    """
    Ensures a datetime object matches the timezone of a pandas Index.
    If the index is tz-aware and the dt is unaware, localizes dt to index tz.
    """
    if not hasattr(index, "tz") or index.tz is None:
        # Index is naive, return naive dt
        if hasattr(dt, "tz") and dt.tz is not None:
            return dt.replace(tzinfo=None)
        return dt
        
    # Index is aware
    if not hasattr(dt, "tz") or dt.tz is None:
        return pd.Timestamp(dt).tz_localize(index.tz)
    
    # Both are aware, convert dt to index tz
    return pd.Timestamp(dt).tz_convert(index.tz)
