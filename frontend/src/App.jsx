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
            <Icon size={20} className="md:w-6 md:h-6" />
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

  useEffect(() => {
    gsap.fromTo(mainRef.current,
      { opacity: 0, y: 30 },
      { opacity: 1, y: 0, duration: 1, ease: "power3.out", delay: 0.2 }
    );
  }, []);

  useEffect(() => {
    loadAll();
  }, [ticker]);

  const loadAll = () => {
    loadMetrics();
    loadPrediction();
    loadPortfolio();
  };

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/api/metrics/${ticker}`);
      if (!response.ok) throw new Error('Failed');
      const data = await response.json();
      setMetrics({
        hybridAcc: data.hybrid_accuracy,
        lstmAcc: data.lstm_accuracy,
        f1: data.f1_score,
        auc: data.auc
      });
    } catch (error) {
      console.error('Error:', error);
      setMetrics({ hybridAcc: 'N/A', lstmAcc: 'N/A', f1: 'N/A', auc: 'N/A' });
    }
    setLoading(false);
  };

  const loadPrediction = async () => {
    try {
      const response = await fetch(`/api/api/predict/${ticker}?threshold=${threshold}`);
      if (!response.ok) throw new Error('Failed');
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const loadPortfolio = async () => {
    try {
      const response = await fetch('/api/api/portfolio');
      if (!response.ok) throw new Error('Failed');
      const data = await response.json();
      setPortfolio(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const executeTrade = async () => {
    if (!prediction || prediction.signal === 'HOLD') return;
    try {
      const response = await fetch(
        `/api/api/trade?ticker=${ticker}&signal=${prediction.signal}&price=${prediction.price}&size=${prediction.suggested_size}`,
        { method: 'POST' }
      );
      if (!response.ok) throw new Error('Failed');
      const data = await response.json();
      setPortfolio(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="min-h-screen bg-base-100 text-base-content relative overflow-hidden font-sans">
      <div className="fixed inset-0 z-[-1] bg-gradient-to-br from-base-300 to-base-100"></div>

      <Sidebar currentTicker={ticker} setTicker={setTicker} />

      <main ref={mainRef} className="ml-0 md:ml-64 p-4 md:p-8 min-h-screen overflow-y-auto w-full md:w-[calc(100%-16rem)]">
        <header className="mb-6 md:mb-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 backdrop-blur-sm bg-base-100/30 p-4 rounded-2xl border border-base-content/5">
          <div>
            <h2 className="text-2xl md:text-4xl font-extrabold pb-2 bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
              {ticker} Dashboard
            </h2>
            <p className="text-sm md:text-base text-base-content/70">Hybrid LSTM & XGBoost Classification</p>
          </div>
          <button onClick={loadAll} className="btn btn-primary btn-sm md:btn-md shadow-lg shadow-primary/30 rounded-full px-6 md:px-8 w-full md:w-auto">
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </header>

        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-6 md:mb-10">
          <MetricCard title="Hybrid Accuracy" value={metrics ? `${metrics.hybridAcc}%` : '...'} icon={Activity} colorClass="text-success" />
          <MetricCard title="LSTM Accuracy" value={metrics ? `${metrics.lstmAcc}%` : '...'} icon={LineChart} colorClass="text-warning" />
          <MetricCard title="F1-Score" value={metrics ? metrics.f1 : '...'} icon={BarChart2} colorClass="text-info" />
          <MetricCard title="ROC AUC" value={metrics ? metrics.auc : '...'} icon={TrendingUp} colorClass="text-secondary" />
        </section>

        <section className="mb-6 md:mb-10">
          <div className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10">
            <div className="card-body">
              <h3 className="card-title text-xl md:text-2xl mb-4 flex items-center gap-2">
                <Activity className="text-primary" /> Live Prediction - {ticker}
              </h3>
              {prediction ? (
                <>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                    <div className="stat bg-base-300 rounded-xl">
                      <div className="stat-title text-xs md:text-sm">Price</div>
                      <div className="stat-value text-primary text-xl md:text-3xl">${prediction.price}</div>
                    </div>
                    <div className="stat bg-base-300 rounded-xl">
                      <div className="stat-title text-xs md:text-sm">Probability</div>
                      <div className="stat-value text-info text-xl md:text-3xl">{prediction.probability}</div>
                    </div>
                    <div className="stat bg-base-300 rounded-xl">
                      <div className="stat-title text-xs md:text-sm">Signal</div>
                      <div className={`stat-value text-xl md:text-3xl ${prediction.signal === 'BUY' ? 'text-success' : prediction.signal === 'SELL' ? 'text-error' : 'text-warning'}`}>
                        {prediction.signal}
                      </div>
                    </div>
                    <div className="stat bg-base-300 rounded-xl">
                      <div className="stat-title text-xs md:text-sm">Suggested Size</div>
                      <div className="stat-value text-secondary text-xl md:text-3xl">{prediction.suggested_size} <span className="text-sm">shares</span></div>
                    </div>
                  </div>
                  {prediction.forecasts && (
                    <div className="mb-4">
                      <h4 className="font-bold mb-2 text-sm md:text-base">Forecast</h4>
                      <div className="overflow-x-auto">
                        <table className="table table-xs md:table-sm">
                          <thead>
                            <tr>
                              <th>Horizon (days)</th>
                              <th>Expected Return</th>
                              <th>Forecast Price</th>
                            </tr>
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
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-4">Loading prediction...</div>
              )}
              {prediction && prediction.signal !== 'HOLD' && (
                <button onClick={executeTrade} className="btn btn-success w-full btn-sm md:btn-md">
                  Execute {prediction.signal} Trade
                </button>
              )}
            </div>
          </div>
        </section>

        <section className="mb-6 md:mb-10">
          <div className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10">
            <div className="card-body">
              <h3 className="card-title text-xl md:text-2xl mb-4 flex items-center gap-2">
                <Wallet className="text-primary" /> Paper Trading Portfolio
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6 mb-6">
                <div className="stat bg-base-300 rounded-xl">
                  <div className="stat-title text-xs md:text-sm">Cash</div>
                  <div className="stat-value text-success text-xl md:text-3xl">${portfolio.cash.toLocaleString()}</div>
                </div>
                <div className="stat bg-base-300 rounded-xl">
                  <div className="stat-title text-xs md:text-sm">Total Equity</div>
                  <div className="stat-value text-primary text-xl md:text-3xl">${portfolio.equity.toLocaleString()}</div>
                </div>
                <div className="stat bg-base-300 rounded-xl">
                  <div className="stat-title text-xs md:text-sm">Open Positions</div>
                  <div className="stat-value text-info text-xl md:text-3xl">{Object.keys(portfolio.positions || {}).length}</div>
                </div>
              </div>
              <div className="flex gap-4 items-center flex-wrap mb-4">
                <label className="flex items-center gap-2 cursor-pointer text-sm md:text-base">
                  <input 
                    type="checkbox" 
                    className="toggle toggle-primary toggle-sm md:toggle-md" 
                    checked={autoTrade}
                    onChange={(e) => setAutoTrade(e.target.checked)}
                  />
                  <span className="font-semibold">Auto-Trade {autoTrade ? <Play size={16} className="inline" /> : <Pause size={16} className="inline" />}</span>
                </label>
                <div className="flex items-center gap-2 text-sm md:text-base">
                  <span className="text-sm">Threshold:</span>
                  <input 
                    type="range" 
                    min="0.5" 
                    max="0.75" 
                    step="0.01" 
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="range range-primary range-xs md:range-sm w-32 md:w-48" 
                  />
                  <span className="badge badge-primary badge-sm md:badge-md">{threshold.toFixed(2)}</span>
                </div>
              </div>
              {portfolio.trades && portfolio.trades.length > 0 && (
                <div className="mt-4">
                  <h4 className="font-bold mb-2 text-sm md:text-base">Recent Trades</h4>
                  <div className="overflow-x-auto">
                    <table className="table table-xs md:table-sm">
                      <thead>
                        <tr>
                          <th>Ticker</th>
                          <th>Action</th>
                          <th>Price</th>
                          <th>Shares</th>
                          <th>Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {portfolio.trades.slice(-5).reverse().map((trade, i) => (
                          <tr key={i}>
                            <td>{trade.ticker}</td>
                            <td className={trade.action === 'BUY' ? 'text-success' : 'text-error'}>{trade.action}</td>
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

        <section className="space-y-6 md:space-y-8 mb-8">
          <div className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10 h-[400px] md:h-[600px]">
            <div className="card-body">
              <h3 className="card-title text-lg md:text-xl mb-4 text-primary">ROC Curve</h3>
              <div className="w-full h-full rounded-xl overflow-hidden bg-base-300/50 flex items-center justify-center">
                <iframe src={`http://localhost:8000/outputs/roc_curve_${ticker}.html`} className="w-full h-full border-0" title="ROC"></iframe>
              </div>
            </div>
          </div>

          <div className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10 h-[400px] md:h-[600px]">
            <div className="card-body">
              <h3 className="card-title text-lg md:text-xl mb-4 text-secondary">Confusion Matrix</h3>
              <div className="w-full h-full rounded-xl overflow-hidden bg-base-300/50 flex items-center justify-center">
                <iframe src={`http://localhost:8000/outputs/confusion_matrix_${ticker}.html`} className="w-full h-full border-0" title="CM"></iframe>
              </div>
            </div>
          </div>

          <div className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10 h-[400px] md:h-[600px]">
            <div className="card-body">
              <h3 className="card-title text-lg md:text-xl mb-4 text-accent">Equity Curve</h3>
              <div className="w-full h-full rounded-xl overflow-hidden bg-base-300/50 flex items-center justify-center">
                <iframe src={`http://localhost:8000/outputs/${ticker}_equity_test.html`} className="w-full h-full border-0" title="Equity"></iframe>
              </div>
            </div>
          </div>

          <div className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10 h-[400px] md:h-[600px]">
            <div className="card-body">
              <h3 className="card-title text-lg md:text-xl mb-4 text-info">Feature Importance</h3>
              <div className="w-full h-full rounded-xl overflow-hidden bg-base-300/50 flex items-center justify-center">
                <iframe src={`http://localhost:8000/outputs/${ticker}_improved_fi.html`} className="w-full h-full border-0" title="FI"></iframe>
              </div>
            </div>
          </div>
        </section>

        <div className="h-20 md:h-32"></div>
      </main>
    </div>
  );
}

export default App;
