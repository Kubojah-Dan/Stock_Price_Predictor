import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { LineChart, Activity, TrendingUp, BarChart2, Wallet, Play, Pause } from 'lucide-react';

function Sidebar({ currentTicker, setTicker }) {
  const [isOpen, setIsOpen] = useState(false);
  const tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA'];
  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="md:hidden fixed top-4 left-4 z-50 btn btn-primary btn-sm"
      >
        {isOpen ? '✕' : '☰'}
      </button>

      <div className={`w-64 bg-base-300 h-screen fixed left-0 top-0 p-4 border-r border-base-content/10 z-40 flex flex-col transition-transform duration-300 ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}`}>
        <h1 className="text-2xl font-bold mb-8 flex items-center gap-2 text-primary mt-12 md:mt-0">
          <Activity size={28} /> Predictor AI
        </h1>
        <div className="flex-1">
          <h2 className="text-sm uppercase text-base-content/50 font-semibold mb-4 tracking-wider">Tickers</h2>
          <ul className="menu bg-base-200 rounded-box p-2">
            {tickers.map(t => (
              <li key={t}>
                <a
                  className={currentTicker === t ? 'active bg-primary text-primary-content' : ''}
                  onClick={() => { setTicker(t); setIsOpen(false); }}
                >
                  <TrendingUp size={16} /> {t}
                </a>
              </li>
            ))}
          </ul>
        </div>
        <div className="text-xs text-base-content/40 text-center mt-auto">
          LSTM-XGBoost Hybrid © 2026
        </div>
      </div>

      {isOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black/50 z-30"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}

function MetricCard({ title, value, icon: Icon, colorClass }) {
  return (
    <div className="card bg-base-200/60 backdrop-blur-md shadow-xl border border-base-content/5 hover:border-primary/50 transition-colors duration-300">
      <div className="card-body p-4 md:p-6">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="text-base-content/60 text-xs md:text-sm font-medium">{title}</h3>
            <div className={`text-2xl md:text-3xl font-bold mt-2 ${colorClass}`}>{value}</div>
          </div>
          <div className={`p-2 md:p-3 rounded-xl bg-base-100 ${colorClass}`}>
            <Icon size={20} />
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [metrics, setMetrics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [portfolio, setPortfolio] = useState({ cash: 100000, equity: 100000, positions: {}, trades: [] });
  const [autoTrade, setAutoTrade] = useState(false);
  const [threshold, setThreshold] = useState(0.60);
  const mainRef = useRef(null);
  const predictionRef = useRef(null);

  useEffect(() => {
    gsap.fromTo(mainRef.current,
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 1, ease: 'power3.out', delay: 0.2 }
    );
  }, []);

  useEffect(() => {
    loadAll();
  }, [ticker]);

  // Keep predictionRef in sync so auto-trade interval can access latest value
  useEffect(() => {
    predictionRef.current = prediction;
  }, [prediction]);

  // Auto-trade interval
  useEffect(() => {
    if (!autoTrade) return;
    const interval = setInterval(async () => {
      const res = await fetch(`/api/predict/${ticker}?threshold=${threshold}`);
      if (!res.ok) return;
      const data = await res.json();
      setPrediction(data);
      if (data.signal !== 'HOLD' && data.suggested_size > 0) {
        const tradeRes = await fetch(
          `/api/trade?ticker=${data.ticker}&signal=${data.signal}&price=${data.price}&size=${data.suggested_size}`,
          { method: 'POST' }
        );
        if (tradeRes.ok) {
          const portfolio = await tradeRes.json();
          setPortfolio(portfolio);
        }
      }
    }, 30000);
    return () => clearInterval(interval);
  }, [autoTrade, ticker, threshold]);

  const loadAll = () => {
    loadMetrics();
    loadPrediction();
    loadPortfolio();
  };

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/metrics/${ticker}`);
      if (!res.ok) throw new Error('Failed');
      const data = await res.json();
      setMetrics({
        hybridAcc: data.hybrid_accuracy,
        lstmAcc: data.lstm_accuracy,
        f1: data.f1_score,
        auc: data.auc,
        precision: data.precision,
        wf_auc: data.wf_auc
      });
    } catch (e) {
      setMetrics({ hybridAcc: 'N/A', lstmAcc: 'N/A', f1: 'N/A', auc: 'N/A', precision: 'N/A', wf_auc: 'N/A' });
    }
    setLoading(false);
  };

  const loadPrediction = async () => {
    try {
      const res = await fetch(`/api/predict/${ticker}?threshold=${threshold}`);
      if (!res.ok) throw new Error('Failed');
      setPrediction(await res.json());
    } catch (e) {
      console.error('Prediction error:', e);
    }
  };

  const loadPortfolio = async () => {
    try {
      const res = await fetch('/api/portfolio');
      if (!res.ok) throw new Error('Failed');
      setPortfolio(await res.json());
    } catch (e) {
      console.error('Portfolio error:', e);
    }
  };

  const executeTrade = async () => {
    if (!prediction || prediction.signal === 'HOLD') return;
    try {
      const res = await fetch(
        `/api/trade?ticker=${ticker}&signal=${prediction.signal}&price=${prediction.price}&size=${prediction.suggested_size}`,
        { method: 'POST' }
      );
      if (!res.ok) throw new Error('Failed');
      setPortfolio(await res.json());
    } catch (e) {
      console.error('Trade error:', e);
    }
  };

  const resetPortfolio = async () => {
    try {
      const res = await fetch('/api/portfolio/reset', { method: 'POST' });
      if (!res.ok) throw new Error('Failed');
      setPortfolio(await res.json());
    } catch (e) {
      console.error('Reset error:', e);
    }
  };

  const pnl = portfolio.equity - 100000;
  const pnlPct = ((pnl / 100000) * 100).toFixed(2);

  return (
    <div className="min-h-screen bg-base-100 text-base-content font-sans">
      <div className="fixed inset-0 z-[-1] bg-gradient-to-br from-base-300 to-base-100"></div>
      <Sidebar currentTicker={ticker} setTicker={setTicker} />

      <main ref={mainRef} className="ml-0 md:ml-64 p-4 md:p-8 min-h-screen w-full md:w-[calc(100%-16rem)]">
        <header className="mb-6 md:mb-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-base-100/30 backdrop-blur-sm p-4 rounded-2xl border border-base-content/5">
          <div>
            <h2 className="text-2xl md:text-4xl font-extrabold pb-2 bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
              {ticker} Dashboard
            </h2>
            <p className="text-sm md:text-base text-base-content/70">Hybrid LSTM & XGBoost Classification</p>
          </div>
          <button onClick={loadAll} className="btn btn-primary btn-sm md:btn-md rounded-full px-6 md:px-8 w-full md:w-auto">
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </header>

        {/* Metrics */}
        <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 md:gap-4 mb-6 md:mb-10">
          <MetricCard title="Accuracy" value={metrics ? `${metrics.hybridAcc}%` : '...'} icon={Activity} colorClass="text-success" />
          <MetricCard title="Bal. Accuracy" value={metrics ? `${metrics.lstmAcc}%` : '...'} icon={LineChart} colorClass="text-warning" />
          <MetricCard title="F1-Score" value={metrics ? metrics.f1 : '...'} icon={BarChart2} colorClass="text-info" />
          <MetricCard title="ROC AUC" value={metrics ? metrics.auc : '...'} icon={TrendingUp} colorClass="text-secondary" />
          <MetricCard title="Precision" value={metrics ? metrics.precision : '...'} icon={Activity} colorClass="text-accent" />
          <MetricCard title="WF AUC" value={metrics ? metrics.wf_auc : '...'} icon={BarChart2} colorClass="text-primary" />
        </section>

        {/* Live Prediction */}
        <section className="mb-6 md:mb-10">
          <div className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10">
            <div className="card-body">
              <h3 className="card-title text-xl md:text-2xl mb-4">
                <Activity className="text-primary" /> Live Prediction — {ticker}
              </h3>
              {prediction ? (
                <>
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                    <div className="stat bg-base-300 rounded-xl">
                      <div className="stat-title text-xs">Price</div>
                      <div className="stat-value text-primary text-xl md:text-2xl">${prediction.price}</div>
                    </div>
                    <div className="stat bg-base-300 rounded-xl">
                      <div className="stat-title text-xs">Probability</div>
                      <div className="stat-value text-info text-xl md:text-2xl">{prediction.probability}</div>
                    </div>
                    <div className="stat bg-base-300 rounded-xl">
                      <div className="stat-title text-xs">Signal</div>
                      <div className={`stat-value text-xl md:text-2xl ${prediction.signal === 'BUY' ? 'text-success' : prediction.signal === 'SELL' ? 'text-error' : 'text-warning'}`}>
                        {prediction.signal}
                      </div>
                    </div>
                    <div className="stat bg-base-300 rounded-xl">
                      <div className="stat-title text-xs">Suggested Size</div>
                      <div className="stat-value text-secondary text-xl md:text-2xl">{prediction.suggested_size} <span className="text-sm">shares</span></div>
                    </div>
                  </div>

                  {prediction.forecasts && (
                    <div className="mb-4 overflow-x-auto">
                      <table className="table table-xs md:table-sm">
                        <thead>
                          <tr><th>Horizon (days)</th><th>Expected Return</th><th>Forecast Price</th></tr>
                        </thead>
                        <tbody>
                          {prediction.forecasts.map((f, i) => (
                            <tr key={i}>
                              <td>{f.days}</td>
                              <td className={f.expected_return >= 0 ? 'text-success' : 'text-error'}>
                                {f.expected_return >= 0 ? '+' : ''}{f.expected_return}%
                              </td>
                              <td>${f.forecast_price}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}

                  {prediction.signal !== 'HOLD' && (
                    <button onClick={executeTrade} className={`btn w-full btn-sm md:btn-md ${prediction.signal === 'BUY' ? 'btn-success' : 'btn-error'}`}>
                      Execute {prediction.signal} — {prediction.suggested_size} shares @ ${prediction.price}
                    </button>
                  )}
                </>
              ) : (
                <div className="text-center py-8 text-base-content/50">Loading prediction...</div>
              )}
            </div>
          </div>
        </section>

        {/* Portfolio */}
        <section className="mb-6 md:mb-10">
          <div className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10">
            <div className="card-body">
              <div className="flex justify-between items-center mb-4">
                <h3 className="card-title text-xl md:text-2xl">
                  <Wallet className="text-primary" /> Paper Trading Portfolio
                </h3>
                <button onClick={resetPortfolio} className="btn btn-error btn-xs md:btn-sm">Reset</button>
              </div>

              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="stat bg-base-300 rounded-xl">
                  <div className="stat-title text-xs">Cash</div>
                  <div className="stat-value text-success text-lg md:text-2xl">${portfolio.cash.toLocaleString(undefined, {maximumFractionDigits: 0})}</div>
                </div>
                <div className="stat bg-base-300 rounded-xl">
                  <div className="stat-title text-xs">Total Equity</div>
                  <div className="stat-value text-primary text-lg md:text-2xl">${portfolio.equity.toLocaleString(undefined, {maximumFractionDigits: 0})}</div>
                </div>
                <div className="stat bg-base-300 rounded-xl">
                  <div className="stat-title text-xs">P&L</div>
                  <div className={`stat-value text-lg md:text-2xl ${pnl >= 0 ? 'text-success' : 'text-error'}`}>
                    {pnl >= 0 ? '+' : ''}${pnl.toLocaleString(undefined, {maximumFractionDigits: 0})}
                  </div>
                  <div className={`stat-desc ${pnl >= 0 ? 'text-success' : 'text-error'}`}>{pnl >= 0 ? '+' : ''}{pnlPct}%</div>
                </div>
                <div className="stat bg-base-300 rounded-xl">
                  <div className="stat-title text-xs">Open Positions</div>
                  <div className="stat-value text-info text-lg md:text-2xl">{Object.keys(portfolio.positions || {}).length}</div>
                </div>
              </div>

              {/* Open Positions Table */}
              {Object.keys(portfolio.positions || {}).length > 0 && (
                <div className="mb-4">
                  <h4 className="font-bold mb-2 text-sm">Open Positions</h4>
                  <div className="overflow-x-auto">
                    <table className="table table-xs md:table-sm">
                      <thead><tr><th>Ticker</th><th>Shares</th><th>Avg Price</th></tr></thead>
                      <tbody>
                        {Object.entries(portfolio.positions).map(([t, pos]) => (
                          <tr key={t}>
                            <td className="font-bold">{t}</td>
                            <td>{pos.shares}</td>
                            <td>${pos.avg_price?.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="flex gap-4 items-center flex-wrap mb-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    className="toggle toggle-primary toggle-sm"
                    checked={autoTrade}
                    onChange={(e) => setAutoTrade(e.target.checked)}
                  />
                  <span className="text-sm font-semibold">
                    Auto-Trade {autoTrade ? <Play size={14} className="inline text-success" /> : <Pause size={14} className="inline" />}
                    {autoTrade && <span className="badge badge-success badge-xs ml-1">LIVE</span>}
                  </span>
                </label>
                <div className="flex items-center gap-2">
                  <span className="text-xs">Threshold:</span>
                  <input
                    type="range" min="0.5" max="0.75" step="0.01"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="range range-primary range-xs w-32 md:w-48"
                  />
                  <span className="badge badge-primary badge-sm">{threshold.toFixed(2)}</span>
                </div>
              </div>

              {portfolio.trades && portfolio.trades.length > 0 && (
                <div>
                  <h4 className="font-bold mb-2 text-sm">Recent Trades</h4>
                  <div className="overflow-x-auto">
                    <table className="table table-xs md:table-sm">
                      <thead><tr><th>Ticker</th><th>Action</th><th>Price</th><th>Shares</th><th>Time</th></tr></thead>
                      <tbody>
                        {portfolio.trades.slice(-5).reverse().map((trade, i) => (
                          <tr key={i}>
                            <td className="font-bold">{trade.ticker}</td>
                            <td className={trade.action === 'BUY' ? 'text-success font-bold' : 'text-error font-bold'}>{trade.action}</td>
                            <td>${trade.price}</td>
                            <td>{trade.shares}</td>
                            <td className="text-xs">{new Date(trade.timestamp).toLocaleTimeString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Visualizations */}
        <section className="space-y-6 md:space-y-8 mb-8">
          {[
            { title: 'ROC Curve', src: `/outputs/${ticker}_roc.html`, color: 'text-primary' },
            { title: 'Confusion Matrix', src: `/outputs/${ticker}_confusion.html`, color: 'text-secondary' },
            { title: 'Equity Curve', src: `/outputs/${ticker}_equity.html`, color: 'text-accent' },
            { title: 'Feature Importance', src: `/outputs/${ticker}_feature_importance.html`, color: 'text-info' },
            { title: 'Drawdown', src: `/outputs/${ticker}_drawdown.html`, color: 'text-error' },
            { title: 'Walk-Forward AUC', src: `/outputs/${ticker}_walk_forward.html`, color: 'text-warning' },
          ].map(({ title, src, color }) => (
            <div key={title} className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10 h-[400px] md:h-[600px]">
              <div className="card-body">
                <h3 className={`card-title text-lg md:text-xl mb-2 ${color}`}>{title}</h3>
                <div className="w-full h-full rounded-xl overflow-hidden bg-base-300/50">
                  <iframe src={src} className="w-full h-full border-0" title={title}></iframe>
                </div>
              </div>
            </div>
          ))}
        </section>

        <div className="h-20 md:h-32"></div>
      </main>
    </div>
  );
}

export default App;
