import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, Shield, TrendingUp, Cpu } from 'lucide-react';

export default function Landing() {
  return (
    <div className="min-h-screen bg-base-100 font-sans">
      <div className="fixed inset-0 z-[-1] bg-gradient-to-br from-base-300 to-base-100"></div>
      
      <header className="p-6 flex justify-between items-center backdrop-blur-md bg-base-100/50 sticky top-0 z-50 border-b border-base-content/5">
        <div className="flex items-center gap-2 text-primary font-bold text-2xl">
          <Activity size={32} /> Predictor AI
        </div>
        <div className="flex gap-4">
          <Link to="/login" className="btn btn-ghost">Login</Link>
          <Link to="/signup" className="btn btn-primary shadow-lg shadow-primary/30 rounded-full px-6">Get Started</Link>
        </div>
      </header>

      <main className="container mx-auto px-4 py-20">
        <div className="text-center max-w-4xl mx-auto mb-20">
          <div className="inline-block mb-4 px-4 py-1.5 rounded-full bg-primary/10 text-primary font-semibold text-sm border border-primary/20">
            Powered by XGBoost & LangChain
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold mb-6 tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-primary via-secondary to-accent">
            Master the Market with AI
          </h1>
          <p className="text-xl text-base-content/70 mb-10 leading-relaxed">
            Connect your XM Demo account. Get real-time predictive analytics on Gold, Tech Stocks, and more. Execute trades automatically or get guided advice from our advanced AI Agent.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link to="/signup" className="btn btn-primary btn-lg rounded-full px-10 shadow-xl shadow-primary/30">
              Start Trading Now
            </Link>
            <Link to="/login" className="btn btn-outline btn-lg rounded-full px-10">
              View Demo
            </Link>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <div className="card bg-base-200/50 backdrop-blur-md border border-base-content/10 hover:border-primary/50 transition-colors">
            <div className="card-body items-center text-center">
              <div className="p-4 bg-primary/20 rounded-2xl text-primary mb-4">
                <TrendingUp size={32} />
              </div>
              <h3 className="card-title text-xl">High-Accuracy Models</h3>
              <p className="text-base-content/70">Our calibrated XGBoost models predict market trends with honest out-of-sample validation.</p>
            </div>
          </div>
          <div className="card bg-base-200/50 backdrop-blur-md border border-base-content/10 hover:border-secondary/50 transition-colors">
            <div className="card-body items-center text-center">
              <div className="p-4 bg-secondary/20 rounded-2xl text-secondary mb-4">
                <Cpu size={32} />
              </div>
              <h3 className="card-title text-xl">AI Agent Advisor</h3>
              <p className="text-base-content/70">Get actionable insights and RAG-powered rationale before you place a trade.</p>
            </div>
          </div>
          <div className="card bg-base-200/50 backdrop-blur-md border border-base-content/10 hover:border-accent/50 transition-colors">
            <div className="card-body items-center text-center">
              <div className="p-4 bg-accent/20 rounded-2xl text-accent mb-4">
                <Shield size={32} />
              </div>
              <h3 className="card-title text-xl">XM Broker Integration</h3>
              <p className="text-base-content/70">Connect seamlessly to your XM Demo account for live paper trading and real-time execution.</p>
            </div>
          </div>
        </div>
      </main>

      <footer className="footer footer-center p-10 bg-base-300 text-base-content mt-20">
        <aside>
          <Activity size={32} className="text-primary mb-2" />
          <p className="font-bold">
            Predictor AI Trading <br/>Providing intelligent trading solutions since 2026
          </p> 
          <p>Copyright © 2026 - All right reserved</p>
        </aside>
      </footer>
    </div>
  );
}
