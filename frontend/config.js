/**
 * ClaimIQ Frontend Configuration
 *
 * LOCAL:  API runs at http://localhost:8000
 * HOSTED: Replace with your Railway URL, e.g. https://claimiq-api.railway.app
 *
 * After deploying to Railway, update RAILWAY_URL below and redeploy Netlify.
 */

const CLAIMIQ_CONFIG = {
  // Update this to your Railway URL after deployment
  API_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : 'https://YOUR-RAILWAY-URL.railway.app',   // ← update after Railway deploy

  VERSION: '1.0.0',
  RULES_COUNT: 14,
};

// Allow override via sessionStorage (for settings page)
const stored = sessionStorage.getItem('claimiq_api');
if (stored) CLAIMIQ_CONFIG.API_URL = stored;

// Make globally available
window.CLAIMIQ_API = CLAIMIQ_CONFIG.API_URL;
