

// ------------------------------------------------------------------
// CONFIGURATION
// ------------------------------------------------------------------
const API_BASE_URL = "https://fairlens-614m.onrender.com/api";

/**
 * Helper: Handle fetch response, throw custom error on failure.
 * @param {Response} response - Fetch Response object
 * @returns {Promise<any>} Parsed JSON response
 * @throws {Error} With status code and message from backend
 */
const handleResponse = async (response) => {
    if (!response.ok) {
        let errorMessage = `Request failed with status ${response.status}`;
        try {
            const errorData = await response.json();
            errorMessage = errorData.error || errorData.message || errorMessage;
        } catch (e) {
            // If response is not JSON, use status text
            errorMessage = response.statusText || errorMessage;
        }
        const error = new Error(errorMessage);
        error.status = response.status;
        throw error;
    }
    return response.json();
};

/**
 * Helper: Create FormData from file and optional fields
 * @param {File} file - CSV file
 * @param {Object} extraFields - Additional fields to append (e.g., targetCol, sensitiveCol)
 * @returns {FormData}
 */
const createFormData = (file, extraFields = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    Object.entries(extraFields).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
            formData.append(key, value);
        }
    });
    return formData;
};

// ------------------------------------------------------------------
// HEALTH & STATUS
// ------------------------------------------------------------------

/**
 * Check if backend is reachable.
 * @returns {Promise<Object>} { status: 'ok', timestamp: string }
 */
export const healthCheck = async () => {
    const response = await fetch(`${API_BASE_URL}/health`);
    return handleResponse(response);
};

// ------------------------------------------------------------------
// COLUMN EXTRACTION
// ------------------------------------------------------------------

/**
 * Upload CSV file and get list of column names.
 * @param {File} file - CSV file
 * @returns {Promise<string[]>} Array of column names
 */
export const getColumns = async (file) => {
    const formData = createFormData(file);
    const response = await fetch(`${API_BASE_URL}/columns`, {
        method: 'POST',
        body: formData,
    });
    const data = await handleResponse(response);
    return data.columns || [];
};

/**
 * Get column names for a built-in sample dataset.
 * @param {string} sampleName - 'adult_income', 'german_credit', or 'compas'
 * @returns {Promise<string[]>} Array of column names
 */
export const getSampleColumns = async (sampleName) => {
    const response = await fetch(`${API_BASE_URL}/sample/${sampleName}/columns`);
    const data = await handleResponse(response);
    return data.columns || [];
};

// ------------------------------------------------------------------
// BIAS ANALYSIS
// ------------------------------------------------------------------

/**
 * Run bias analysis on uploaded CSV file.
 * @param {File} file - CSV file
 * @param {string} targetCol - Name of target column (binary outcome)
 * @param {string} sensitiveCol - Name of sensitive attribute column
 * @returns {Promise<Object>} Bias analysis result (see bias_detector.py output)
 */
export const analyze = async (file, targetCol, sensitiveCol) => {
    const formData = createFormData(file, { targetCol, sensitiveCol });
    const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
    });
    return handleResponse(response);
};

/**
 * Run bias analysis on built-in sample dataset.
 * @param {string} sampleName - 'adult_income', 'german_credit', 'compas'
 * @param {string} targetCol - Target column name
 * @param {string} sensitiveCol - Sensitive attribute column name
 * @returns {Promise<Object>} Bias analysis result
 */
export const analyzeSample = async (sampleName, targetCol, sensitiveCol) => {
    const url = new URL(`${API_BASE_URL}/sample/${sampleName}/analyze`);
    url.searchParams.append('targetCol', targetCol);
    url.searchParams.append('sensitiveCol', sensitiveCol);
    const response = await fetch(url.toString());
    return handleResponse(response);
};

// ------------------------------------------------------------------
// SHAP EXPLANATIONS
// ------------------------------------------------------------------

/**
 * Get SHAP explanations for uploaded CSV model.
 * @param {File} file - CSV file
 * @param {string} targetCol - Target column name
 * @param {string} sensitiveCol - Sensitive attribute column name
 * @returns {Promise<Object>} SHAP result (top_features, per_group_shap, explanation)
 */
export const getExplanation = async (file, targetCol, sensitiveCol) => {
    const formData = createFormData(file, { targetCol, sensitiveCol });
    const response = await fetch(`${API_BASE_URL}/explain`, {
        method: 'POST',
        body: formData,
    });
    return handleResponse(response);
};

/**
 * Get SHAP explanations for built-in sample dataset.
 * @param {string} sampleName - 'adult_income', 'german_credit', 'compas'
 * @param {string} targetCol - Target column name
 * @param {string} sensitiveCol - Sensitive attribute column name
 * @returns {Promise<Object>} SHAP result
 */
export const getExplanationSample = async (sampleName, targetCol, sensitiveCol) => {
    const url = new URL(`${API_BASE_URL}/sample/${sampleName}/explain`);
    url.searchParams.append('targetCol', targetCol);
    url.searchParams.append('sensitiveCol', sensitiveCol);
    const response = await fetch(url.toString());
    return handleResponse(response);
};

// ------------------------------------------------------------------
// FAIRNESS MITIGATION
// ------------------------------------------------------------------

/**
 * Apply fairness mitigation (ExponentiatedGradient) to uploaded CSV model.
 * @param {File} file - CSV file
 * @param {string} targetCol - Target column name
 * @param {string} sensitiveCol - Sensitive attribute column name
 * @returns {Promise<Object>} Fair model result (fair_accuracy, fair_dp_diff, etc.)
 */
export const mitigate = async (file, targetCol, sensitiveCol) => {
    const formData = createFormData(file, { targetCol, sensitiveCol });
    const response = await fetch(`${API_BASE_URL}/mitigate`, {
        method: 'POST',
        body: formData,
    });
    return handleResponse(response);
};

/**
 * Apply fairness mitigation to built-in sample dataset.
 * @param {string} sampleName - 'adult_income', 'german_credit', 'compas'
 * @param {string} targetCol - Target column name
 * @param {string} sensitiveCol - Sensitive attribute column name
 * @returns {Promise<Object>} Fair model result
 */
export const mitigateSample = async (sampleName, targetCol, sensitiveCol) => {
    const url = new URL(`${API_BASE_URL}/sample/${sampleName}/mitigate`);
    url.searchParams.append('targetCol', targetCol);
    url.searchParams.append('sensitiveCol', sensitiveCol);
    const response = await fetch(url.toString());
    return handleResponse(response);
};

// ------------------------------------------------------------------
// CONVENIENCE: AUTO-DETECT FILE OR SAMPLE
// ------------------------------------------------------------------

/**
 * Universal explanation fetcher – works with either file or sample.
 * @param {File|string} fileOrSample - File object or sample name string
 * @param {string} targetCol - Target column
 * @param {string} sensitiveCol - Sensitive column
 * @param {boolean} isSample - True if second param is sample name
 * @returns {Promise<Object>}
 */
export const getExplanationAuto = async (fileOrSample, targetCol, sensitiveCol, isSample = false) => {
    if (isSample) {
        return getExplanationSample(fileOrSample, targetCol, sensitiveCol);
    } else {
        return getExplanation(fileOrSample, targetCol, sensitiveCol);
    }
};

/**
 * Universal mitigation fetcher.
 * @param {File|string} fileOrSample - File object or sample name
 * @param {string} targetCol - Target column
 * @param {string} sensitiveCol - Sensitive column
 * @param {boolean} isSample - True if first param is sample name
 * @returns {Promise<Object>}
 */
export const mitigateAuto = async (fileOrSample, targetCol, sensitiveCol, isSample = false) => {
    if (isSample) {
        return mitigateSample(fileOrSample, targetCol, sensitiveCol);
    } else {
        return mitigate(fileOrSample, targetCol, sensitiveCol);
    }
};

// ------------------------------------------------------------------
// EXPORT ALL
// ------------------------------------------------------------------
export default {
    healthCheck,
    getColumns,
    getSampleColumns,
    analyze,
    analyzeSample,
    getExplanation,
    getExplanationSample,
    getExplanationAuto,
    mitigate,
    mitigateSample,
    mitigateAuto,
};