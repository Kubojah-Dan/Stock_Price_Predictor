import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import gsap from 'gsap';
import { LineChart, Activity, TrendingUp, BarChart2, Wallet, Play, Pause, LogOut, Link as LinkIcon, Bot } from 'lucide-react';

function Sidebar({ currentTicker, setTicker, onLogout }) {
  const [isOpen, setIsOpen] = useState(false);
  const tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'GC=F'];
  return (
    <>
      <button onClick={() => setIsOpen(!isOpen)} className="md:hidden fixed top-4 left-4 z-50 btn btn-primary btn-sm">
        {isOpen ? '✕' : '☰'}
      </button>

      <div className={`w-64 bg-base-300 h-screen fixed left-0 top-0 p-4 border-r border-base-content/10 z-40 flex flex-col transition-transform duration-300 ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}`}>
        <h1 className="text-2xl font-bold mb-8 flex items-center gap-2 text-primary mt-12 md:mt-0">
          <Activity size={28} /> Predictor AI
        </h1>
        <div className="flex-1">
          <h2 className="text-sm uppercase text-base-content/50 font-semibold mb-4 tracking-wider">Markets</h2>
          <ul className="menu bg-base-200 rounded-box p-2">
            {tickers.map(t => (
              <li key={t}>
                <a className={currentTicker === t ? 'active bg-primary text-primary-content' : ''} onClick={() => { setTicker(t); setIsOpen(false); }}>
                  <TrendingUp size={16} /> {t}
                </a>
              </li>
            ))}
          </ul>
        </div>
        <div className="mt-auto flex flex-col gap-4">
          <button onClick={onLogout} className="btn btn-outline btn-error btn-sm w-full"><LogOut size={16} /> Logout</button>
          <div className="text-xs text-base-content/40 text-center">
            LSTM-XGBoost + LLM © 2026
          </div>
        </div>
      </div>

      {isOpen && <div className="md:hidden fixed inset-0 bg-black/50 z-30" onClick={() => setIsOpen(false)} />}
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
            <div className={`text-xl md:text-2xl font-bold mt-2 ${colorClass}`}>{value}</div>
          </div>
          <div className={`p-2 md:p-3 rounded-xl bg-base-100 ${colorClass}`}><Icon size={20} /></div>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [ticker, setTicker] = useState('GC=F');
  const [metrics, setMetrics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [agentReasoning, setAgentReasoning] = useState('');
  const [loadingAgent, setLoadingAgent] = useState(false);
  
  const [brokerStatus, setBrokerStatus] = useState(null);
  const [xmForm, setXmForm] = useState({ account: '', password: '', server: 'XMGlobal-Demo' });
  const [showXmModal, setShowXmModal] = useState(false);

  const [loading, setLoading] = useState(false);
  const [threshold, setThreshold] = useState(0.60);
  const navigate = useNavigate();
  const mainRef = useRef(null);

  const token = localStorage.getItem('token');
  const authHeaders = { 'Authorization': `Bearer ${token}` };

  function handleLogout() {
    localStorage.removeItem('token');
    navigate('/login');
  }

  useEffect(() => {
    gsap.fromTo(mainRef.current, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 1, ease: 'power3.out', delay: 0.2 });
    loadBrokerStatus();
  }, []);

  const [trendUrl, setTrendUrl] = useState('');

  useEffect(() => {
    loadAll();
    loadTrend();
  }, [ticker]);

  const loadTrend = async () => {
    try {
      const res = await fetch(`http://localhost:8000/api/plots/trend/${ticker}`);
      const data = await res.json();
      setTrendUrl(data.plot_url);
    } catch (e) {
      console.error("Trend load error", e);
    }
  };

  const loadAll = () => {
    loadMetrics();
    loadPrediction();
  };

  const loadBrokerStatus = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/broker/status', { headers: authHeaders });
      const data = await res.json();
      setBrokerStatus(data);
    } catch (e) {
      console.error('Broker status error', e);
    }
  };

  const connectXM = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/user/xm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders },
        body: JSON.stringify(xmForm)
      });
      if (!res.ok) throw new Error("Failed to connect");
      setShowXmModal(false);
      // Wait a bit for MT5 to init
      setTimeout(loadBrokerStatus, 2000);
    } catch (e) {
      alert("Failed to connect to MT5. Ensure MetaTrader 5 terminal is running on the server machine.");
    }
    setLoading(false);
  };

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/api/metrics/${ticker}`);
      if (!res.ok) throw new Error('Failed');
      setMetrics(await res.json());
    } catch (e) {
      setMetrics({ hybrid_accuracy: 'N/A', lstm_accuracy: 'N/A', f1_score: 'N/A', auc: 'N/A', precision: 'N/A', threshold: '0.5' });
    }
    setLoading(false);
  };

  const loadPrediction = async () => {
    try {
      const res = await fetch(`http://localhost:8000/api/predict/${ticker}?threshold=${threshold}`);
      if (!res.ok) throw new Error('Failed');
      setPrediction(await res.json());
      setAgentReasoning(''); // clear previous reasoning when ticker changes
    } catch (e) {
      console.error('Prediction error:', e);
    }
  };

  const askAgent = async () => {
    setLoadingAgent(true);
    try {
      const res = await fetch(`http://localhost:8000/api/agent/chat/${ticker}?threshold=${threshold}`, { headers: authHeaders });
      if (!res.ok) throw new Error('Agent failed');
      const data = await res.json();
      setAgentReasoning(data.decision);
    } catch (e) {
      setAgentReasoning('Agent is unavailable. Ensure Groq API key is configured correctly.');
    }
    setLoadingAgent(false);
  };

  return (
    <div className="min-h-screen bg-base-100 text-base-content font-sans">
      <div className="fixed inset-0 z-[-1] bg-gradient-to-br from-base-300 to-base-100"></div>
      <Sidebar currentTicker={ticker} setTicker={setTicker} onLogout={handleLogout} />

      <main ref={mainRef} className="ml-0 md:ml-64 p-4 md:p-8 min-h-screen w-full md:w-[calc(100%-16rem)]">
        <header className="mb-6 md:mb-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-base-100/30 backdrop-blur-sm p-4 rounded-2xl border border-base-content/5 shadow-lg">
          <div>
            <h2 className="text-2xl md:text-4xl font-extrabold pb-2 bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
              {ticker} Overview
            </h2>
            <p className="text-sm md:text-base text-base-content/70">ML Classification + AI RAG Agent</p>
          </div>
          <div className="flex gap-4 w-full md:w-auto">
            {brokerStatus?.status === 'connected' ? (
              <div className="badge badge-success badge-outline p-4 font-bold flex gap-2">
                <LinkIcon size={16}/> XM Connected
              </div>
            ) : (
              <button onClick={() => setShowXmModal(true)} className="btn btn-outline btn-secondary btn-sm md:btn-md flex-1 md:flex-none">
                <LinkIcon size={16}/> Connect XM
              </button>
            )}
            <button onClick={loadAll} className="btn btn-primary btn-sm md:btn-md rounded-full px-6 flex-1 md:flex-none">
              {loading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        </header>

        {/* XM Broker Modal */}
        {showXmModal && (
          <div className="modal modal-open">
            <div className="modal-box">
              <h3 className="font-bold text-lg mb-4">Connect XM Demo Account</h3>
              <form onSubmit={connectXM}>
                <div className="form-control mb-2">
                  <label className="label">Account ID</label>
                  <input type="text" className="input input-bordered" value={xmForm.account} onChange={e=>setXmForm({...xmForm, account: e.target.value})} required/>
                </div>
                <div className="form-control mb-2">
                  <label className="label">Password</label>
                  <input type="password" className="input input-bordered" value={xmForm.password} onChange={e=>setXmForm({...xmForm, password: e.target.value})} required/>
                </div>
                <div className="form-control mb-6">
                  <label className="label">Server</label>
                  <input type="text" className="input input-bordered" value={xmForm.server} onChange={e=>setXmForm({...xmForm, server: e.target.value})} required/>
                </div>
                <div className="flex justify-end gap-2">
                  <button type="button" onClick={() => setShowXmModal(false)} className="btn btn-ghost">Cancel</button>
                  <button type="submit" className="btn btn-primary" disabled={loading}>{loading ? 'Connecting...' : 'Connect MT5'}</button>
                </div>
              </form>
            </div>
            <div className="modal-backdrop" onClick={() => setShowXmModal(false)}></div>
          </div>
        )}

        {/* Broker Summary if connected */}
        {brokerStatus?.status === 'connected' && brokerStatus.info && (
          <section className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div className="stat bg-success/10 rounded-2xl border border-success/20">
              <div className="stat-title text-success/80">XM Balance</div>
              <div className="stat-value text-success">${brokerStatus.info.balance.toFixed(2)}</div>
              <div className="stat-desc text-success/70">Login: {brokerStatus.info.login}</div>
            </div>
            <div className="stat bg-success/10 rounded-2xl border border-success/20">
              <div className="stat-title text-success/80">XM Equity</div>
              <div className="stat-value text-success">${brokerStatus.info.equity.toFixed(2)}</div>
            </div>
            <div className="stat bg-success/10 rounded-2xl border border-success/20">
              <div className="stat-title text-success/80">Free Margin</div>
              <div className="stat-value text-success">${brokerStatus.info.free_margin.toFixed(2)}</div>
            </div>
            <div className="stat bg-success/10 rounded-2xl border border-success/20">
              <div className="stat-title text-success/80">Open Positions</div>
              <div className="stat-value text-success">{brokerStatus.positions?.length || 0}</div>
            </div>
          </section>
        )}

        <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 md:gap-4 mb-6 md:mb-10">
          <MetricCard title="Accuracy" value={metrics ? `${metrics.hybrid_accuracy}%` : '...'} icon={Activity} colorClass="text-success" />
          <MetricCard title="Bal. Accuracy" value={metrics ? `${metrics.lstm_accuracy}%` : '...'} icon={LineChart} colorClass="text-warning" />
          <MetricCard title="F1-Score" value={metrics ? metrics.f1_score : '...'} icon={BarChart2} colorClass="text-info" />
          <MetricCard title="ROC AUC" value={metrics ? metrics.auc : '...'} icon={TrendingUp} colorClass="text-secondary" />
          <MetricCard title="Precision" value={metrics ? metrics.precision : '...'} icon={Activity} colorClass="text-accent" />
          <MetricCard title="Threshold" value={metrics ? metrics.threshold : '...'} icon={BarChart2} colorClass="text-primary" />
        </section>

        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Prediction & ML Stats */}
          <div className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10 shadow-xl">
            <div className="card-body">
              <h3 className="card-title text-xl md:text-2xl mb-4 border-b border-base-content/10 pb-4">
                <Activity className="text-primary" /> Live Prediction
              </h3>
              
              <div className="flex items-center gap-2 mb-6">
                <span className="text-sm font-semibold">Threshold:</span>
                <input type="range" min="0.5" max="0.75" step="0.01" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} className="range range-primary range-xs flex-1" />
                <span className="badge badge-primary">{threshold.toFixed(2)}</span>
              </div>

              {prediction ? (
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-base-300 p-4 rounded-xl text-center">
                    <div className="text-xs font-semibold opacity-60 uppercase">Price</div>
                    <div className="text-2xl font-bold text-primary mt-1">${prediction.price}</div>
                  </div>
                  <div className="bg-base-300 p-4 rounded-xl text-center">
                    <div className="text-xs font-semibold opacity-60 uppercase">ML Probability</div>
                    <div className="text-2xl font-bold text-info mt-1">{prediction.probability}</div>
                  </div>
                  <div className="bg-base-300 p-4 rounded-xl text-center col-span-2">
                    <div className="text-xs font-semibold opacity-60 uppercase">Signal generated</div>
                    <div className={`text-4xl font-extrabold mt-1 ${prediction.signal === 'BUY' ? 'text-success' : prediction.signal === 'SELL' ? 'text-error' : 'text-warning'}`}>
                      {prediction.signal}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-10 opacity-50">Loading prediction...</div>
              )}
            </div>
          </div>

          {/* AI Advisor Panel */}
          <div className="card bg-gradient-to-br from-base-200/80 to-base-300/80 backdrop-blur-xl border border-primary/20 shadow-2xl shadow-primary/5">
            <div className="card-body">
              <h3 className="card-title text-xl md:text-2xl mb-4 border-b border-base-content/10 pb-4 flex justify-between">
                <div className="flex items-center gap-2 text-secondary"><Bot /> AI Trading Advisor</div>
                <button onClick={askAgent} disabled={loadingAgent || !prediction} className="btn btn-secondary btn-sm rounded-full">
                  {loadingAgent ? <span className="loading loading-spinner loading-xs"></span> : 'Analyze'}
                </button>
              </h3>

              <div className="flex-1 bg-base-100/50 rounded-xl p-4 overflow-y-auto min-h-[200px] text-sm leading-relaxed whitespace-pre-wrap">
                {agentReasoning ? (
                  agentReasoning
                ) : (
                  <div className="text-center opacity-40 mt-10">
                    Click analyze to get RAG-powered reasoning from the AI agent based on the latest ML predictions and market context.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Forecasts Section */}
        {prediction && prediction.forecasts && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            {prediction.forecasts.map((f, i) => (
              <div key={i} className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10 shadow-lg hover:border-primary/50 transition-all duration-300">
                <div className="card-body p-4 flex flex-row items-center justify-between">
                  <div>
                    <h4 className="text-xs uppercase font-bold opacity-50 tracking-wider">{f.days} Day Forecast</h4>
                    <div className="text-2xl font-black mt-1 text-primary">${f.forecast_price}</div>
                  </div>
                  <div className={`text-right ${f.expected_return >= 0 ? 'text-success' : 'text-error'}`}>
                    <div className="text-lg font-bold flex items-center justify-end gap-1">
                      {f.expected_return >= 0 ? '▲' : '▼'} {Math.abs(f.expected_return)}%
                    </div>
                    <div className="text-[10px] opacity-60 uppercase font-bold">est. return</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Trend Visualization */}
        {trendUrl && (
          <div className="card bg-base-200/50 backdrop-blur-xl border border-primary/20 h-[500px] mb-8 overflow-hidden shadow-2xl">
            <div className="card-body p-0">
               <iframe src={trendUrl} className="w-full h-full border-0" title="Recent Trend"></iframe>
            </div>
          </div>
        )}

        {/* Visualizations */}
        <section className="space-y-6 md:space-y-8 mb-8">
          {[
            { title: 'ROC Curve', src: `/outputs/${ticker}_roc.html`, color: 'text-primary' },
            { title: 'Confusion Matrix', src: `/outputs/${ticker}_cm.html`, color: 'text-secondary' },
            { title: 'Equity Curve', src: `/outputs/${ticker}_equity.html`, color: 'text-accent' },
            { title: 'Feature Importance', src: `/outputs/${ticker}_fi.html`, color: 'text-success' },
          ].map(({ title, src, color }) => (
            <div key={title} className="card bg-base-200/50 backdrop-blur-xl border border-base-content/10 h-[400px] md:h-[600px]">
              <div className="card-body">
                <h3 className={`card-title text-lg md:text-xl mb-2 ${color}`}>{title}</h3>
                <div className="w-full h-full rounded-xl overflow-hidden bg-base-300/50">
                  <iframe src={`http://localhost:8000${src}`} className="w-full h-full border-0" title={title}></iframe>
                </div>
              </div>
            </div>
          ))}
        </section>

      </main>
    </div>
  );
}
