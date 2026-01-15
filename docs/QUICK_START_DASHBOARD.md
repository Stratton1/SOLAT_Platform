# SOLAT Dashboard - Quick Start Guide

## 🚀 Starting the Dashboard

### Step 1: Start the Backend (Sentinel)
```bash
cd /Users/Joe/Desktop/SOLAT_Platform
python3 run_sentinel.py
```

Expected output:
```
✓ Database initialized
✓ Assets loaded from seed
Starting event loop (interval=60s)...
```

### Step 2: Start the Frontend (Dashboard)
In another terminal:
```bash
cd /Users/Joe/Desktop/SOLAT_Platform
python3 run_dashboard.py
```

Expected output:
```
✓ Streamlit 1.50.0 found
✓ Database found at data/db/trading_engine.db
Launching Streamlit dashboard...
→ Opening http://localhost:8501 in browser
```

### Step 3: Access the Dashboard
Navigate to: **http://localhost:8501**

---

## 📊 Dashboard Tabs Overview

### Tab 1: 📊 Live Surveillance
**What it shows:**
- Live ticker of top 3 assets by fitness
- 4 KPI metric cards:
  - 📊 Total Assets (count + active breakdown)
  - 🎯 Active Signals (Buy/Sell split)
  - 💪 Portfolio Fitness (average score)
  - 💓 Sentinel Heartbeat (live/recent/stale status)
- Full market surveillance table (all assets)
- Interactive Ichimoku chart (select asset)
- Strategy explanation guide (expandable)

**Key Features:**
- ✅ Color-coded signals (🟢 BUY, 🔴 SELL, ⚪ NEUTRAL)
- ✅ Hover effects on cards and tables
- ✅ Live ticker updates every 30 seconds
- ✅ Detailed strategy explanation with examples

### Tab 2: 📈 Evolution & Fitness
**What it shows:**
- Fitness ranking bar chart (top 5 assets)
- Status distribution pie chart
- Asset status metrics (Active/Normal/Dormant)
- Full asset table with fitness scores
- Fitness calculation formula

**Key Features:**
- ✅ Visual fitness rankings
- ✅ Status distribution breakdown
- ✅ Sortable asset table
- ✅ Fitness formula explanation

### Tab 3: 📝 Trades & Performance
**What it shows:**
- Paper trading summary metrics
- Recent trades table (last 20)
- Entry/exit prices and P&L
- Paper trading mode explanation

**Key Features:**
- ✅ Trade history view
- ✅ P&L calculations
- ✅ Safe paper trading mode
- ✅ No real capital at risk

### Tab 4: 🔧 System Health
**What it shows:**
- System configuration (scan intervals, risk per trade)
- System info (status, mode, database type)
- Ichimoku strategy parameters (9/26/52 periods)
- System debug information
- Raw data viewer (advanced)

**Key Features:**
- ✅ Full configuration view
- ✅ Strategy parameter reference
- ✅ Advanced raw data explorer
- ✅ System health metrics

---

## 🧪 Backtesting Page

### Access Backtesting
The backtesting feature is in a **separate page** (accessed from Streamlit's multi-page feature).

If not showing as a separate tab:
```
Note: Backtesting is in dashboard/pages/backtest.py
Run the dashboard, then look for additional pages in the sidebar
```

### Using the Backtester
1. **Sidebar Controls:**
   - Select Asset: Choose from BTC/USDT, ETH/USDT, AAPL, GC=F
   - Time Period: Select start and end dates
   - Initial Capital: Enter your starting balance

2. **Run Backtest:**
   - Click "🚀 Run Backtest" button
   - Wait for simulation to complete

3. **Review Results:**
   - View equity curve
   - Check key metrics (Return%, MaxDD%, Sharpe, WinRate%)
   - Review trade history
   - Read analysis and recommendations

---

## 🎨 UI Elements Explained

### Metric Cards
```
┌─────────────────────────┐
│ 📊 Total Assets         │
│ 4                       │
│ 3 Active                │
└─────────────────────────┘
```
- Shows key performance indicators
- Hover effects with border color change
- Delta showing additional context

### Live Ticker
```
BTC/USDT    $47,523.00    0.823 🟢 BUY
ETH/USDT    $2,845.15     0.612 ⚪ NEUTRAL
AAPL        $185.42       0.445 🔴 SELL
```
- Top 3 assets by fitness score
- Live prices and signal status
- Updates with market snapshots

### Surveillance Table
```
┌────────┬──────┬────────┬────────┬─────────┐
│ Asset  │ Source │ Status │ Fitness │ Signal  │
├────────┼──────┼────────┼────────┼─────────┤
│ BTC    │ ccxt   │ Active │ 0.823  │ 🟢 BUY  │
│ ETH    │ ccxt   │ Normal │ 0.612  │ ⚪ N/A  │
└────────┴──────┴────────┴────────┴─────────┘
```
- Color-coded rows (green=active, orange=normal, red=dormant)
- Sticky headers when scrolling
- Hover effects on rows

### Status Badges
```
🟢 Active (Top 20%)        Green
🟡 Normal (Middle 60%)     Orange
🔴 Dormant (Bottom 20%)    Red
```
- Indicate asset priority level
- Active = scanned every 5 min
- Dormant = scanned every 1 hour

### Signal Colors
```
🟢 BUY:     Price > Cloud AND Tenkan > Kijun
🔴 SELL:    Price < Cloud AND Tenkan < Kijun
⚪ NEUTRAL: No clear signal
```

---

## 📊 Understanding the Metrics

### Fitness Score
```
Fitness = (0.4 × Win Rate) + (0.4 × Profit Factor) - (0.2 × Max Drawdown)
```
- **Win Rate**: % of profitable trades
- **Profit Factor**: Total Wins ÷ Total Losses
- **Max Drawdown**: Largest peak-to-trough decline

**Example:**
- Win Rate: 60% = 0.60
- Profit Factor: 3.0 = 3.0
- Max Drawdown: 20% = 0.20
- Fitness = (0.4 × 0.60) + (0.4 × 3.0) - (0.2 × 0.20)
- Fitness = 0.24 + 1.20 - 0.04 = **1.40**

### Ichimoku Cloud Components

**Tenkan-sen (Blue Line)**
- 9-period momentum indicator
- Fast-moving line
- Crosses above Kijun = bullish signal

**Kijun-sen (Red Line)**
- 26-period momentum indicator
- Medium-term support/resistance
- Crosses below Tenkan = bearish signal

**Cloud (Senkou Spans A & B)**
- 🟩 **Green Cloud**: Senkou A > B (Bullish)
- 🟥 **Red Cloud**: Senkou A < B (Bearish)
- Price > Cloud: Uptrend
- Price < Cloud: Downtrend

**Chikou Span (Purple Dashed)**
- Current close shifted 26 periods back
- Confirms trend strength
- Above price = bullish
- Below price = bearish

---

## 🔧 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `R` | Refresh page |
| `S` | Focus search/sidebar |
| `?` | Show Streamlit help |
| `Ctrl+C` | Stop dashboard (in terminal) |

---

## 🐛 Troubleshooting

### Issue: "Waiting for first market scan..."
**Solution:** Ensure Sentinel is running in Terminal 1
```bash
# In Terminal 1:
python3 run_sentinel.py
# Should show: "Starting event loop..."
```

### Issue: Dashboard won't load
**Solution:** Check if port 8501 is in use
```bash
lsof -i :8501  # Find what's using port
python3 run_dashboard.py --server.port=8502  # Use different port
```

### Issue: CSS styling not loading
**Solution:** The style.css file might be in wrong location
```bash
ls dashboard/assets/style.css
# Should exist and have 450+ lines
```

### Issue: Charts not rendering
**Solution:** Ensure plotly is installed
```bash
pip install plotly
python3 run_dashboard.py  # Try again
```

---

## 📈 Advanced Tips

### Filtering Data
In the **Live Surveillance** tab, use the sidebar to filter:
- ✅ **Show Only Active Assets**: See top 20% by fitness
- ✅ **Show Only Buy/Sell Signals**: Hide NEUTRAL signals

### Viewing Raw Data
In the **System Health** tab, expand **"Raw Data Viewer (Advanced)"**:
- View all market snapshots
- Inspect asset data
- Review complete trade history

### Backtest Different Scenarios
In the **Backtester** page:
1. Try different assets
2. Test different time periods
3. Adjust initial capital
4. Compare equity curves

### Monitoring Heartbeat
In the **Live Surveillance** tab, watch the **Sentinel Heartbeat** card:
- 🟢 **Live**: Updated in last 2 minutes
- 🟡 **Recent**: Updated in last 10 minutes
- 🔴 **Stale**: No update in 10+ minutes

---

## 🎯 Navigation Guide

```
Dashboard Home
├─ Sidebar
│  ├─ System Status
│  ├─ Filters (Active Assets, Buy/Sell Signals)
│  ├─ Signal Legend
│  └─ About SOLAT
│
├─ Tab 1: Live Surveillance
│  ├─ Live Ticker (top 3 assets)
│  ├─ KPI Metrics (4 cards)
│  ├─ Surveillance Table (all assets)
│  ├─ Ichimoku Chart (select asset)
│  └─ Strategy Explanation (expandable)
│
├─ Tab 2: Evolution & Fitness
│  ├─ Fitness Bar Chart (top 5)
│  ├─ Status Pie Chart
│  ├─ Status Metrics (3 cards)
│  └─ Full Asset Table
│
├─ Tab 3: Trades & Performance
│  ├─ Paper Trading Summary
│  ├─ Recent Trades Table
│  └─ Trade Details
│
└─ Tab 4: System Health
   ├─ Configuration
   ├─ System Info
   ├─ Strategy Parameters
   ├─ Debug Information
   └─ Raw Data Viewer
```

---

## ⏱️ Auto-Refresh Schedule

- **Dashboard**: Updates every 30 seconds
- **Sentinel**: Scans every 60 seconds
- **Evolution**: Recalculates every 4 hours
- **Charts**: Load on-demand (when selected)

---

## 📱 Mobile/Tablet View

The dashboard is responsive:
- **Desktop (1200px+)**: 4-column layouts
- **Tablet (768-1200px)**: 2-column layouts
- **Mobile (<768px)**: 1-column stacked layout

---

## 🎓 Learning Resources

### Inside the Dashboard
- **Strategy Explanation**: Tab 1 → Bottom (expandable)
- **Fitness Calculation**: Tab 2 → Status Distribution section
- **System Parameters**: Tab 4 → Ichimoku Strategy Parameters

### External Resources
- **Ichimoku Cloud**: https://en.wikipedia.org/wiki/Ichimoku_Kink%C5%8D_Hy%C5%8D
- **SOLAT Architecture**: See `CLAUDE.md` in project root
- **Dashboard Upgrade Guide**: See `DASHBOARD_UPGRADE.md`

---

## 🎉 You're Ready!

The SOLAT Dashboard is now ready for use. Enjoy your pro-grade trading terminal!

**Key Features Available:**
- ✅ Live market monitoring
- ✅ Ichimoku Cloud analysis
- ✅ Fitness-based asset ranking
- ✅ Interactive backtesting
- ✅ Paper trading simulation
- ✅ Professional dark theme
- ✅ 24/7 autonomous operation

**Questions?** Check the strategy explanation guide in Tab 1, or review `DASHBOARD_UPGRADE.md` for detailed documentation.
