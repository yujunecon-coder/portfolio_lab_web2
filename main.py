import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
from io import BytesIO

st.set_page_config(page_title="Interactive Portfolio Lab", layout="wide")
st.title("Interactive Portfolio Lab")
st.write("Interactive tool for teaching Mean-Variance Portfolio, CAPM, and Efficient Frontier")

# -----------------------------
# 1. 股票列表和选择
# -----------------------------
stock_options = {
    "Tech": ["AAPL","MSFT","NVDA","GOOGL","META"],
    "Banking": ["JPM","BAC","C","WFC"],
    "Energy": ["XOM","CVX","BP"],
    "Consumer": ["PG","KO","PEP"]
}

st.subheader("Select Stocks")
selected_sectors = st.multiselect("Choose sectors", list(stock_options.keys()), default=["Tech"])
selected_stocks = []

for sector in selected_sectors:
    sector_stocks = stock_options[sector]
    sector_selected = st.multiselect(f"{sector} stocks", sector_stocks, default=sector_stocks)
    selected_stocks += sector_selected

if not selected_stocks:
    st.warning("Please select at least one stock.")
    st.stop()

# -----------------------------
# 2. 日期和风险自由率输入
# -----------------------------
st.subheader("Select Time Period and Risk Free Rate")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", pd.to_datetime("2019-01-01"))
with col2:
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
rf = st.slider("Risk free rate (%)", 0.0, 10.0, 2.0)/100

# -----------------------------
# 3. 下载数据和计算收益率（安全版）
# -----------------------------
data_raw = yf.download(selected_stocks, start=start_date, end=end_date, group_by='ticker')

# 检查返回的列名类型
if isinstance(data_raw.columns, pd.MultiIndex):
    # 多只股票
    # yfinance有时候多股返回 MultiIndex: 第一层是股票代码，第二层是字段
    adj_close_dict = {}
    for stock in selected_stocks:
        try:
            adj_close_dict[stock] = data_raw[stock]['Adj Close']
        except KeyError:
            # 如果没有 'Adj Close'，用 'Close' 替代
            adj_close_dict[stock] = data_raw[stock]['Close']
    data = pd.DataFrame(adj_close_dict)
else:
    # 单只股票
    if 'Adj Close' in data_raw.columns:
        data = data_raw['Adj Close'].to_frame()
    else:
        # 如果没有 'Adj Close'，用 'Close' 替代
        data = data_raw['Close'].to_frame()

# 计算收益率
returns = data.pct_change().dropna()
mu = returns.mean() * 252
Sigma = returns.cov() * 252

st.subheader("Mean Returns (Annualized)")
st.dataframe(mu)
st.subheader("Covariance Matrix")
st.dataframe(Sigma)

# -----------------------------
# 4. Portfolio functions
# -----------------------------
n = len(selected_stocks)
w0 = np.ones(n)/n
bounds = [(0,1)]*n
constraints = ({'type':'eq','fun':lambda w: np.sum(w)-1})

def portfolio_return(w):
    return np.dot(w, mu)

def portfolio_vol(w):
    return np.sqrt(np.dot(w.T, np.dot(Sigma, w)))

# -----------------------------
# 5. Minimum Variance Portfolio
# -----------------------------
min_var_res = minimize(portfolio_vol, w0, bounds=bounds, constraints=constraints)
w_min = min_var_res.x
st.subheader("Minimum Variance Portfolio Weights")
st.dataframe(pd.DataFrame({"Stock":selected_stocks, "Weight":w_min}))

# -----------------------------
# 6. Efficient Frontier
# -----------------------------
def min_vol_target(target):
    cons = (
        {'type':'eq','fun':lambda w: np.sum(w)-1},
        {'type':'eq','fun':lambda w: portfolio_return(w)-target}
    )
    res = minimize(portfolio_vol, w0, bounds=bounds, constraints=cons)
    return res.x

target_returns = np.linspace(mu.min(), mu.max(), 40)
vols = [portfolio_vol(min_vol_target(r)) for r in target_returns]

# Monte Carlo Portfolio Cloud
N = 3000
mc_returns = []
mc_vols = []
for i in range(N):
    w = np.random.random(n)
    w /= np.sum(w)
    mc_returns.append(portfolio_return(w))
    mc_vols.append(portfolio_vol(w))

# -----------------------------
# 7. Tangency Portfolio
# -----------------------------
def neg_sharpe(w):
    return -(portfolio_return(w)-rf)/portfolio_vol(w)

tan_res = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints)
w_tan = tan_res.x
ret_tan = portfolio_return(w_tan)
vol_tan = portfolio_vol(w_tan)
sharpe_tan = (ret_tan-rf)/vol_tan
st.subheader("Tangency Portfolio Weights and Sharpe Ratio")
st.dataframe(pd.DataFrame({"Stock":selected_stocks, "Weight":w_tan}))
st.write(f"Sharpe Ratio: {sharpe_tan:.3f}")

# -----------------------------
# 8. Plot Efficient Frontier and Capital Market Line
# -----------------------------
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(mc_vols, mc_returns, c='gray', alpha=0.2, label="Random Portfolios")
ax.plot(vols, target_returns, 'r-', linewidth=3, label="Efficient Frontier")
ax.scatter(vol_tan, ret_tan, marker="*", s=200, c='gold', label="Tangency Portfolio")
cml_x = [0, vol_tan*1.5]
cml_y = [rf, rf + sharpe_tan*(vol_tan*1.5)]
ax.plot(cml_x, cml_y, 'b--', linewidth=2, label="Capital Market Line")
ax.set_xlabel("Volatility")
ax.set_ylabel("Return")
ax.set_title("Efficient Frontier and Capital Market Line")
ax.legend()
st.pyplot(fig)

# -----------------------------
# 9. CAPM Beta Estimation
# -----------------------------
st.subheader("CAPM Beta Estimation")
market_ticker = st.selectbox("Select Market Index", ["^GSPC","^IXIC","^DJI"], index=0)
stock_for_beta = st.selectbox("Select Stock for Beta Estimation", selected_stocks)

# -----------------------------
# CAPM Beta Estimation 安全版
# -----------------------------
market_data_raw = yf.download(market_ticker, start=start_date, end=end_date)

# 检查列名类型
if isinstance(market_data_raw.columns, pd.MultiIndex):
    # 多层索引
    if 'Adj Close' in market_data_raw.columns.get_level_values(1):
        market_data = market_data_raw.xs('Adj Close', axis=1, level=1)
    elif 'Close' in market_data_raw.columns.get_level_values(1):
        market_data = market_data_raw.xs('Close', axis=1, level=1)
    else:
        # 如果没有 Adj Close 或 Close，直接取第一个列
        market_data = market_data_raw.iloc[:,0]
else:
    # 单层索引
    if 'Adj Close' in market_data_raw.columns:
        market_data = market_data_raw['Adj Close']
    elif 'Close' in market_data_raw.columns:
        market_data = market_data_raw['Close']
    else:
        # 如果没有 Adj Close 或 Close，直接取第一个列
        market_data = market_data_raw.iloc[:,0]

# 计算收益率
market_returns = market_data.pct_change().dropna()
stock_returns = returns[stock_for_beta].loc[market_returns.index]

# CAPM回归
X = sm.add_constant(market_returns)
model = sm.OLS(stock_returns, X).fit()
beta = model.params[1]
alpha = model.params[0]
st.write(f"Stock: {stock_for_beta}, Beta: {beta:.3f}, Alpha: {alpha:.3f}")

# Security Market Line plot
fig2, ax2 = plt.subplots(figsize=(8,5))
sml_x = np.array([0, beta*1.5])
sml_y = rf + beta*(market_returns.mean()*252 - rf)/beta*sml_x
ax2.plot(sml_x, sml_y, 'r-', label="Security Market Line")
ax2.scatter(beta, portfolio_return(w0), c='blue', label=f"{stock_for_beta}")
ax2.set_xlabel("Beta")
ax2.set_ylabel("Expected Return")
ax2.set_title("Security Market Line")
ax2.legend()
st.pyplot(fig2)

# -----------------------------
# 10. Download Excel
# -----------------------------
st.subheader("Download Portfolio Data as Excel")
output_df = pd.DataFrame({"Stock":selected_stocks, 
                          "MinimumVariance":w_min, 
                          "Tangency":w_tan})
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    output_df.to_excel(writer, index=False, sheet_name="Portfolio Weights")
st.download_button("Download Excel", buffer, file_name="portfolio_weights.xlsx", mime="application/vnd.ms-excel")

# -----------------------------
# 11. Interactive Portfolio Weight Slider
# -----------------------------
st.subheader("Interactive Portfolio Weight Adjustment")
st.write("Drag sliders to adjust weights. Total weight is automatically normalized to 1.")

w_slider = []
cols = st.columns(len(selected_stocks))
for i, stock in enumerate(selected_stocks):
    with cols[i]:
        w = st.slider(f"{stock} weight", 0.0, 1.0, float(1.0/n), step=0.01, key=f"w{i}")
        w_slider.append(w)

w_slider = np.array(w_slider)
if np.sum(w_slider) == 0:
    w_slider = np.ones(n)/n
else:
    w_slider = w_slider / np.sum(w_slider)

interactive_ret = portfolio_return(w_slider)
interactive_vol = portfolio_vol(w_slider)
st.write(f"Portfolio Expected Return: {interactive_ret:.3f}")
st.write(f"Portfolio Volatility: {interactive_vol:.3f}")

fig3, ax3 = plt.subplots(figsize=(10,6))
ax3.scatter(mc_vols, mc_returns, c='gray', alpha=0.2, label="Random Portfolios")
ax3.plot(vols, target_returns, 'r-', linewidth=3, label="Efficient Frontier")
ax3.scatter(vol_tan, ret_tan, marker="*", s=200, c='gold', label="Tangency Portfolio")
ax3.scatter(interactive_vol, interactive_ret, c='blue', s=150, marker="o", label="Your Portfolio")
ax3.set_xlabel("Volatility")
ax3.set_ylabel("Return")
ax3.set_title("Efficient Frontier with Interactive Portfolio")
ax3.legend()
st.pyplot(fig3)
st.dataframe(pd.DataFrame({"Stock":selected_stocks, "Interactive Weight":w_slider}))