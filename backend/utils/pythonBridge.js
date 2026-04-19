

const fetch = require('node-fetch');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

// ------------------------------------------------------------------
// CONFIGURATION
// ------------------------------------------------------------------
const FLASK_URL = process.env.FLASK_URL || 'https://fairlens-614m.onrender.com';
const API_PREFIX = '/api';

/**
 * Custom error class for Flask communication errors.
 * Includes HTTP status code from Flask if available.
 */
class FlaskError extends Error {
    constructor(message, statusCode = 500) {
        super(message);
        this.name = 'FlaskError';
        this.statusCode = statusCode;
    }
}

// ------------------------------------------------------------------
// HELPER: Handle Flask response
// ------------------------------------------------------------------
/**
 * Process the fetch response, parse JSON, and throw on error.
 * @param {Response} response - Fetch Response object
 * @returns {Promise<Object>} Parsed JSON response body
 * @throws {FlaskError} If response is not ok or parsing fails
 */
const handleFlaskResponse = async (response) => {
    let data;
    try {
        data = await response.json();
    } catch (err) {
        throw new FlaskError(`Invalid response from Flask: ${err.message}`, response.status);
    }

    if (!response.ok) {
        const errorMsg = data.error || data.message || `Flask request failed with status ${response.status}`;
        throw new FlaskError(errorMsg, response.status);
    }

    return data;
};

// ------------------------------------------------------------------
// HELPER: Create multipart form with file and fields
// ------------------------------------------------------------------
/**
 * Builds a FormData object containing a CSV file and optional text fields.
 * @param {string} filePath - Path to CSV file on disk
 * @param {Object} fields - Key-value pairs to append as text fields
 * @returns {FormData} Ready-to-send FormData instance
 */
const buildFormData = (filePath, fields = {}) => {
    const form = new FormData();
    // Append file stream
    const readStream = fs.createReadStream(filePath);
    form.append('file', readStream, { filename: path.basename(filePath) });
    // Append additional fields
    Object.entries(fields).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
            form.append(key, value);
        }
    });
    return form;
};

// ------------------------------------------------------------------
// PUBLIC FUNCTIONS
// ------------------------------------------------------------------

/**
 * Check if Flask ML engine is reachable.
 * @returns {Promise<Object>} Health status object
 */
const healthCheck = async () => {
    const url = `${FLASK_URL}${API_PREFIX}/health`;
    const response = await fetch(url);
    return handleFlaskResponse(response);
};

/**
 * Retrieve column names from a CSV file using Flask.
 * @param {string} filePath - Path to uploaded CSV file
 * @returns {Promise<string[]>} Array of column names
 */
const getColumns = async (filePath) => {
    const form = buildFormData(filePath);
    const url = `${FLASK_URL}${API_PREFIX}/columns`;

    const response = await fetch(url, {
        method: 'POST',
        body: form,
        headers: form.getHeaders()
    });

    const data = await handleFlaskResponse(response);
    return data.columns || [];
};

/**
 * Retrieve column names from a built-in sample dataset.
 * @param {string} datasetName - Sample dataset name
 * @returns {Promise<string[]>} Array of column names
 */
const getSampleColumns = async (datasetName) => {
    const url = `${FLASK_URL}${API_PREFIX}/sample/${datasetName}/columns`;
    const response = await fetch(url);
    const data = await handleFlaskResponse(response);
    return data.columns || [];
};

/**
 * Run full bias analysis on a CSV file.
 * @param {string} filePath - Path to CSV file
 * @param {string} targetCol - Name of target column
 * @param {string} sensitiveCol - Name of sensitive attribute column
 * @returns {Promise<Object>} Bias analysis result (see bias_detector.py output)
 */
const analyze = async (filePath, targetCol, sensitiveCol) => {
    const form = buildFormData(filePath, { targetCol, sensitiveCol });
    const url = `${FLASK_URL}${API_PREFIX}/analyze`;

    const response = await fetch(url, {
        method: 'POST',
        body: form,
        headers: form.getHeaders()
    });

    return handleFlaskResponse(response);
};

/**
 * Run bias analysis on a built-in sample dataset (no file upload).
 * @param {string} datasetName - One of: 'adult_income', 'german_credit', 'compas'
 * @param {string} targetCol - Target column name
 * @param {string} sensitiveCol - Sensitive column name
 * @returns {Promise<Object>} Bias analysis result
 */
const sampleAnalysis = async (datasetName, targetCol, sensitiveCol) => {
    const url = new URL(`${FLASK_URL}${API_PREFIX}/sample/${datasetName}/analyze`);
    url.searchParams.append('targetCol', targetCol);
    url.searchParams.append('sensitiveCol', sensitiveCol);

    const response = await fetch(url.toString());
    return handleFlaskResponse(response);
};

/**
 * Get SHAP explanations for a CSV file.
 * @param {string} filePath - Path to CSV file
 * @param {string} targetCol - Target column name
 * @param {string} sensitiveCol - Sensitive column name
 * @returns {Promise<Object>} SHAP result (top_features, per_group_shap, explanation)
 */
const getExplanation = async (filePath, targetCol, sensitiveCol) => {
    const form = buildFormData(filePath, { targetCol, sensitiveCol });
    const url = `${FLASK_URL}${API_PREFIX}/explain`;

    const response = await fetch(url, {
        method: 'POST',
        body: form,
        headers: form.getHeaders()
    });

    return handleFlaskResponse(response);
};

/**
 * Get SHAP explanations for a built-in sample dataset.
 * @param {string} datasetName - Sample dataset name
 * @param {string} targetCol - Target column name
 * @param {string} sensitiveCol - Sensitive column name
 * @returns {Promise<Object>} SHAP result
 */
const getSampleExplanation = async (datasetName, targetCol, sensitiveCol) => {
    const url = new URL(`${FLASK_URL}${API_PREFIX}/sample/${datasetName}/explain`);
    url.searchParams.append('targetCol', targetCol);
    url.searchParams.append('sensitiveCol', sensitiveCol);

    const response = await fetch(url.toString());
    return handleFlaskResponse(response);
};

/**
 * Apply fairness mitigation (ExponentiatedGradient) to a CSV file.
 * @param {string} filePath - Path to CSV file
 * @param {string} targetCol - Target column name
 * @param {string} sensitiveCol - Sensitive column name
 * @returns {Promise<Object>} Fair model result (fair_accuracy, fair_dp_diff, etc.)
 */
const mitigate = async (filePath, targetCol, sensitiveCol) => {
    const form = buildFormData(filePath, { targetCol, sensitiveCol });
    const url = `${FLASK_URL}${API_PREFIX}/mitigate`;

    const response = await fetch(url, {
        method: 'POST',
        body: form,
        headers: form.getHeaders()
    });

    return handleFlaskResponse(response);
};

/**
 * Apply fairness mitigation to a built-in sample dataset.
 * @param {string} datasetName - Sample dataset name
 * @param {string} targetCol - Target column name
 * @param {string} sensitiveCol - Sensitive column name
 * @returns {Promise<Object>} Fair model result
 */
const mitigateSample = async (datasetName, targetCol, sensitiveCol) => {
    const url = new URL(`${FLASK_URL}${API_PREFIX}/sample/${datasetName}/mitigate`);
    url.searchParams.append('targetCol', targetCol);
    url.searchParams.append('sensitiveCol', sensitiveCol);

    const response = await fetch(url.toString());
    return handleFlaskResponse(response);
};

// ------------------------------------------------------------------
// EXPORTS
// ------------------------------------------------------------------
module.exports = {
    healthCheck,
    getColumns,
    getSampleColumns,
    analyze,
    sampleAnalysis,
    getExplanation,
    getSampleExplanation,
    mitigate,
    mitigateSample,
    FlaskError
};