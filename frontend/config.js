/**
 * ClaimIQ Frontend Configuration
 */
const MEDFLOW_CONFIG = {
  API_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : 'https://healthcare-claims-agent-production-a15d.up.railway.app',
};

// Allow override via sessionStorage (settings page)
const _stored = sessionStorage.getItem('medflow_api');
if (_stored) MEDFLOW_CONFIG.API_URL = _stored;

window.MEDFLOW_API = MEDFLOW_CONFIG.API_URL;