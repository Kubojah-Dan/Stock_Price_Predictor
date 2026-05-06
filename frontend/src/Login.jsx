import React, { useState } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import { LogIn } from 'lucide-react';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [totpCode, setTotpCode] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const res = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, totp_code: totpCode })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Login failed');
      
      // Save token and navigate to dashboard
      localStorage.setItem('token', data.access_token);
      navigate('/dashboard');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="min-h-screen bg-base-300 flex items-center justify-center p-4">
      <div className="card w-full max-w-md bg-base-100 shadow-2xl border border-base-content/10">
        <div className="card-body">
          <div className="flex justify-center mb-6">
            <div className="p-3 bg-primary/20 rounded-2xl text-primary">
              <LogIn size={32} />
            </div>
          </div>
          
          <h2 className="text-2xl font-bold text-center mb-6">Welcome Back</h2>

          {location.state?.message && (
            <div className="alert alert-success text-sm mb-4">{location.state.message}</div>
          )}
          {error && <div className="alert alert-error text-sm mb-4">{error}</div>}

          <form onSubmit={handleLogin}>
            <div className="form-control mb-4">
              <label className="label"><span className="label-text">Email</span></label>
              <input 
                type="email" 
                className="input input-bordered w-full" 
                value={email}
                onChange={e => setEmail(e.target.value)}
                required 
              />
            </div>
            <div className="form-control mb-4">
              <label className="label"><span className="label-text">Password</span></label>
              <input 
                type="password" 
                className="input input-bordered w-full" 
                value={password}
                onChange={e => setPassword(e.target.value)}
                required 
              />
            </div>
            <div className="form-control mb-6">
              <label className="label">
                <span className="label-text">2FA Authenticator Code</span>
              </label>
              <input 
                type="text" 
                className="input input-bordered w-full tracking-widest text-center font-mono" 
                maxLength={6}
                value={totpCode}
                onChange={e => setTotpCode(e.target.value)}
                required 
              />
            </div>
            <button className="btn btn-primary w-full shadow-lg shadow-primary/30">Login</button>
            <div className="text-center mt-4 text-sm">
              Don't have an account? <Link to="/signup" className="text-primary font-bold hover:underline">Sign Up</Link>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
