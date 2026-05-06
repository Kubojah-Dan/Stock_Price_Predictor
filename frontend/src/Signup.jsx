import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { ShieldCheck, Activity } from 'lucide-react';

export default function Signup() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [qrUri, setQrUri] = useState('');
  const [totpCode, setTotpCode] = useState('');
  const [step, setStep] = useState(1);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const res = await fetch('http://localhost:8000/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Registration failed');
      
      // Transform URI to Google Chart API QR Code image URL for easy rendering
      const qrImageUrl = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(data.qr_uri)}`;
      setQrUri(qrImageUrl);
      setStep(2);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleVerify2FA = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const res = await fetch('http://localhost:8000/api/auth/verify-2fa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, totp_code: totpCode })
      });
      if (!res.ok) throw new Error('Invalid 2FA code');
      
      // Navigate to login after successful setup
      navigate('/login', { state: { message: 'Registration complete. Please login.' } });
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
              {step === 1 ? <Activity size={32} /> : <ShieldCheck size={32} />}
            </div>
          </div>
          
          <h2 className="text-2xl font-bold text-center mb-6">
            {step === 1 ? 'Create Account' : 'Set Up 2FA'}
          </h2>

          {error && <div className="alert alert-error text-sm mb-4">{error}</div>}

          {step === 1 ? (
            <form onSubmit={handleRegister}>
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
              <div className="form-control mb-6">
                <label className="label"><span className="label-text">Password</span></label>
                <input 
                  type="password" 
                  className="input input-bordered w-full" 
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required 
                />
              </div>
              <button className="btn btn-primary w-full shadow-lg shadow-primary/30">Continue</button>
              <div className="text-center mt-4 text-sm">
                Already have an account? <Link to="/login" className="text-primary font-bold hover:underline">Login</Link>
              </div>
            </form>
          ) : (
            <form onSubmit={handleVerify2FA} className="text-center">
              <p className="text-sm mb-4">Scan this QR code with your authenticator app (like Google Authenticator or Authy).</p>
              <div className="bg-white p-4 rounded-xl inline-block mb-6 shadow-sm">
                <img src={qrUri} alt="2FA QR Code" className="w-48 h-48" />
              </div>
              <div className="form-control mb-6 text-left">
                <label className="label"><span className="label-text">6-digit Code</span></label>
                <input 
                  type="text" 
                  className="input input-bordered w-full text-center text-xl tracking-widest font-mono" 
                  maxLength={6}
                  value={totpCode}
                  onChange={e => setTotpCode(e.target.value)}
                  required 
                />
              </div>
              <button className="btn btn-primary w-full shadow-lg shadow-primary/30">Verify & Complete</button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
