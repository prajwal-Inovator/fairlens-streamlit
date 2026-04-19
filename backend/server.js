

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');

// ------------------------------------------------------------------
// ROUTE IMPORTS
// ------------------------------------------------------------------
const uploadRoutes = require('./routes/upload');
const analyzeRoutes = require('./routes/analyze');
const resultsRoutes = require('./routes/results');
// Additional routes to be implemented:
const explainRoutes = require('./routes/explain');
const mitigateRoutes = require('./routes/mitigate');
const sampleRoutes = require('./routes/sample');

// ------------------------------------------------------------------
// INITIALIZATION
// ------------------------------------------------------------------
const app = express();
const PORT = process.env.PORT || 4000;
const NODE_ENV = process.env.NODE_ENV || 'development';

// ------------------------------------------------------------------
// MIDDLEWARE
// ------------------------------------------------------------------
// Security headers
app.use(helmet({
    crossOriginResourcePolicy: { policy: "cross-origin" }
}));

// CORS configuration – allow React frontend (localhost:3000)
app.use(cors({
    origin: ['http://localhost:3000', 'http://127.0.0.1:3000', 'https://fairlens-frontend.onrender.com'],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

// Request logging
if (NODE_ENV === 'development') {
    app.use(morgan('dev'));
} else {
    app.use(morgan('combined'));
}

// Parse JSON request bodies (for non-file routes)
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const sampleRoutes = require('./routes/sample');
app.use('/api', sampleRoutes);

// ------------------------------------------------------------------
// CREATE TEMP DIRECTORY IF NOT EXISTS
// ------------------------------------------------------------------
const tempDir = process.env.TEMP_DIR || '/tmp/fairlens_uploads';
if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
    console.log(`[✓] Created temp directory: ${tempDir}`);
}
// Override temp dir for multer (will be used by route handlers)
process.env.TEMP_DIR = tempDir;

// ------------------------------------------------------------------
// HEALTH CHECK ENDPOINT (direct, not in a separate route file)
// ------------------------------------------------------------------
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        service: 'fairlens-backend',
        version: '1.0.0'
    });
});

// ------------------------------------------------------------------
// MOUNT ROUTES
// ------------------------------------------------------------------
app.use('/api', uploadRoutes);     // POST /api/columns, /api/validate
app.use('/api', analyzeRoutes);    // POST /api/analyze
app.use('/api/results', resultsRoutes); // CRUD for stored results

// Placeholder for future routes (commented until implemented)
app.use('/api', explainRoutes);   // POST /api/explain
app.use('/api', mitigateRoutes);  // POST /api/mitigate
app.use('/api/sample', sampleRoutes); // GET /api/sample/:name/...

// ------------------------------------------------------------------
// 404 HANDLER (for unmatched routes)
// ------------------------------------------------------------------
app.use((req, res) => {
    res.status(404).json({ error: `Route ${req.method} ${req.url} not found` });
});

// ------------------------------------------------------------------
// GLOBAL ERROR HANDLER
// ------------------------------------------------------------------
app.use((err, req, res, next) => {
    console.error('Unhandled error:', err);

    // Handle multer errors specifically
    if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({ error: 'File too large. Maximum size is 100MB.' });
    }
    if (err.code === 'LIMIT_UNEXPECTED_FILE') {
        return res.status(400).json({ error: 'Unexpected field name. Use "file" as the field name.' });
    }

    // Default error response
    const statusCode = err.statusCode || 500;
    const message = err.message || 'Internal server error';
    res.status(statusCode).json({ error: message });
});

// ------------------------------------------------------------------
// START SERVER
// ------------------------------------------------------------------
app.listen(PORT, () => {
    console.log(`
╔══════════════════════════════════════════════════════════════╗
║                    FAIRLENS BACKEND SERVER                   ║
╠══════════════════════════════════════════════════════════════╣
║  Port: ${PORT}                                               ║
║  Environment: ${NODE_ENV.padEnd(40)}                         ║
║  Temp directory: ${tempDir.padEnd(36)}                       ║
║  Flask URL: ${process.env.FLASK_URL || 'http://localhost:5000'}${' '.repeat(35 - (process.env.FLASK_URL || 'http://localhost:5000').length)}║
╚══════════════════════════════════════════════════════════════╝
  `);
});

// ------------------------------------------------------------------
// GRACEFUL SHUTDOWN (clean up temp files on exit)
// ------------------------------------------------------------------
process.on('SIGINT', () => {
    console.log('\n[✓] Shutting down gracefully...');
    // Optionally clean entire temp directory (be careful)
    // fs.rmSync(tempDir, { recursive: true, force: true });
    process.exit(0);
});

module.exports = app;