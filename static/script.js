// ── Stores last prediction result for PDF on predict page ──
window._lastPrediction = null;

// ══════════════════════════════════════════════════════════════
//  SHARED: generateMedicalPDF(data)
//  Called from both predict.html and dashboard.html modal
//  data = { date, result, prob, age, sex, cp, trestbps,
//           chol, fbs, restecg, thalach, exang, oldpeak,
//           slope, ca, thal }
// ══════════════════════════════════════════════════════════════
function generateMedicalPDF(d) {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const W = 210, H = 297;

    // ── Header bar
    doc.setFillColor(15, 23, 42);
    doc.rect(0, 0, W, 42, 'F');
    doc.setFillColor(59, 130, 246);
    doc.rect(0, 42, W, 3, 'F');

    doc.setFontSize(22); doc.setFont('helvetica', 'bold'); doc.setTextColor(255, 255, 255);
    doc.text('Heart Disease AI Portal', 15, 18);
    doc.setFontSize(10); doc.setFont('helvetica', 'normal'); doc.setTextColor(148, 163, 184);
    doc.text('AI-Powered Cardiac Risk Assessment Report', 15, 27);
    doc.text('Generated: ' + new Date().toLocaleString(), 15, 34);
    doc.text('Report Date: ' + d.date, 110, 34);

    // ── Diagnosis Result
    let y = 55;
    doc.setFontSize(13); doc.setFont('helvetica', 'bold'); doc.setTextColor(59, 130, 246);
    doc.text('DIAGNOSIS RESULT', 15, y);
    doc.setDrawColor(59, 130, 246); doc.setLineWidth(0.5); doc.line(15, y + 2, W - 15, y + 2);
    y += 10;

    const isHigh = d.result === 1 && d.prob >= 70;
    const isMed  = d.result === 1 && d.prob < 70;
    const riskLabel = isHigh ? 'HIGH RISK' : isMed ? 'MODERATE RISK' : 'LOW RISK';
    const [rR, rG, rB] = isHigh ? [239, 68, 68] : isMed ? [245, 158, 11] : [16, 185, 129];

    doc.setFillColor(rR, rG, rB);
    doc.roundedRect(15, y, W - 30, 18, 3, 3, 'F');
    doc.setFontSize(14); doc.setFont('helvetica', 'bold'); doc.setTextColor(255, 255, 255);
    doc.text(riskLabel + '  —  ' + d.prob + '% Heart Disease Risk Probability', W / 2, y + 12, { align: 'center' });
    y += 26;

    // ── Clinical Vitals table
    const CP_L    = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'];
    const THAL_L  = { 0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect', 3: 'Unknown' };
    const SLOPE_L = { 0: 'Upsloping', 1: 'Flat', 2: 'Downsloping' };
    const ECG_L   = { 0: 'Normal', 1: 'ST Abnormality', 2: 'LV Hypertrophy' };

    doc.setFontSize(13); doc.setFont('helvetica', 'bold'); doc.setTextColor(59, 130, 246);
    doc.text('CLINICAL VITALS & INPUT PARAMETERS', 15, y);
    doc.setDrawColor(59, 130, 246); doc.line(15, y + 2, W - 15, y + 2);
    y += 10;

    const rows = [
        ['Age',             d.age + ' years',              'Sex',                    d.sex === 1 ? 'Male' : 'Female'],
        ['Chest Pain Type', CP_L[d.cp]    || d.cp,         'Resting Blood Pressure', d.trestbps + ' mmHg'],
        ['Cholesterol',     d.chol + ' mg/dl',             'Fasting Blood Sugar>120', d.fbs === 1 ? 'Yes' : 'No'],
        ['Resting ECG',     ECG_L[d.restecg] || d.restecg, 'Max Heart Rate',          d.thalach + ' BPM'],
        ['Exercise Angina', d.exang === 1 ? 'Yes' : 'No',  'ST Depression',           String(d.oldpeak)],
        ['ST Slope',        SLOPE_L[d.slope] || d.slope,   'Major Vessels',           String(d.ca)],
        ['Thalassemia',     THAL_L[d.thal]   || d.thal,    '',                        ''],
    ];

    rows.forEach((row, i) => {
        if (i % 2 === 0) { doc.setFillColor(240, 245, 255); doc.rect(15, y - 4, W - 30, 12, 'F'); }
        doc.setFontSize(9); doc.setFont('helvetica', 'bold'); doc.setTextColor(30, 64, 175);
        doc.text(row[0], 18, y + 3);
        doc.setFont('helvetica', 'normal'); doc.setTextColor(30, 30, 30);
        doc.text(String(row[1]), 68, y + 3);
        if (row[2]) {
            doc.setFont('helvetica', 'bold'); doc.setTextColor(30, 64, 175);
            doc.text(row[2], 115, y + 3);
            doc.setFont('helvetica', 'normal'); doc.setTextColor(30, 30, 30);
            doc.text(String(row[3]), 170, y + 3);
        }
        y += 12;
    });
    y += 6;

    // ── AI Assessment
    doc.setFontSize(13); doc.setFont('helvetica', 'bold'); doc.setTextColor(59, 130, 246);
    doc.text('AI ASSESSMENT SUMMARY', 15, y);
    doc.setDrawColor(59, 130, 246); doc.line(15, y + 2, W - 15, y + 2);
    y += 10;

    const summary = d.result === 1
        ? `The AI model has detected indicators consistent with elevated cardiac risk. The risk probability of ${d.prob}% suggests that further medical evaluation is strongly recommended. Early intervention and lifestyle modifications can significantly improve outcomes.`
        : `The AI model has not detected significant indicators of heart disease. The risk probability of ${d.prob}% falls within the lower risk category. Continue maintaining a healthy lifestyle and schedule regular check-ups as a preventive measure.`;

    doc.setFontSize(10); doc.setFont('helvetica', 'normal'); doc.setTextColor(50, 50, 50);
    const summaryLines = doc.splitTextToSize(summary, W - 30);
    doc.text(summaryLines, 15, y);
    y += summaryLines.length * 5 + 8;

    // ── Recommendations
    doc.setFontSize(13); doc.setFont('helvetica', 'bold'); doc.setTextColor(59, 130, 246);
    doc.text('RECOMMENDATIONS', 15, y);
    doc.setDrawColor(59, 130, 246); doc.line(15, y + 2, W - 15, y + 2);
    y += 10;

    const recs = d.result === 1 ? [
        '• Consult a cardiologist or healthcare provider for a comprehensive evaluation.',
        '• Monitor blood pressure and cholesterol levels regularly.',
        '• Adopt a heart-healthy diet (reduce saturated fats, sodium, and sugar).',
        '• Engage in moderate physical activity as advised by your physician.',
        '• Avoid smoking and limit alcohol consumption.',
        '• Follow up with ECG, stress tests, or imaging as recommended.',
    ] : [
        '• Maintain a balanced diet rich in fruits, vegetables, and whole grains.',
        '• Engage in at least 150 minutes of moderate exercise per week.',
        '• Schedule annual health check-ups including blood pressure and cholesterol.',
        '• Avoid smoking and limit alcohol consumption.',
        '• Manage stress through mindfulness or relaxation techniques.',
    ];

    doc.setFontSize(10); doc.setFont('helvetica', 'normal'); doc.setTextColor(50, 50, 50);
    recs.forEach(r => { doc.text(r, 15, y); y += 7; });

    // ── Footer
    doc.setFillColor(15, 23, 42);
    doc.rect(0, H - 22, W, 22, 'F');
    doc.setFontSize(8); doc.setFont('helvetica', 'italic'); doc.setTextColor(148, 163, 184);
    doc.text('MEDICAL DISCLAIMER: This report is generated by an AI tool for educational purposes only.', W / 2, H - 14, { align: 'center' });
    doc.text('It does not constitute medical advice. Always consult a qualified healthcare professional.', W / 2, H - 9, { align: 'center' });
    doc.setFont('helvetica', 'normal'); doc.setTextColor(100, 116, 139);
    doc.text('Page 1 of 1  |  Heart Disease AI Portal', W / 2, H - 4, { align: 'center' });

    const filename = 'heart-report-' + String(d.date || 'report').replace(/[:/,\s]/g, '-') + '.pdf';
    doc.save(filename);
}

// ══════════════════════════════════════════════════════════════
//  PREDICT PAGE — Form submit handler
// ══════════════════════════════════════════════════════════════
const form = document.getElementById('prediction-form');

if (form) {
    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        const ekgContainer       = document.getElementById('ekg-container');
        const resultSection      = document.getElementById('result-section');
        const probFill           = document.getElementById('prob-fill');
        const probText           = document.getElementById('prob-text');
        const resultTitle        = document.getElementById('result-title');
        const resultDesc         = document.getElementById('result-description');
        const insightsContainer  = document.getElementById('insights-container');
        const resultActions      = document.getElementById('result-actions');

        form.classList.add('hidden');
        resultSection.classList.add('hidden');
        ekgContainer.classList.remove('hidden');
        if (typeof startMessages === 'function') startMessages();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();

            setTimeout(() => {
                if (typeof stopMessages === 'function') stopMessages();
                ekgContainer.classList.add('hidden');

                if (result.success) {
                    const probPercent = (result.probability * 100).toFixed(1);

                    // Save globally so downloadPDF() on predict page can use it
                    window._lastPrediction = {
                        date:     new Date().toLocaleString(),
                        result:   result.prediction,
                        prob:     parseFloat(probPercent),
                        age:      parseFloat(data.age)      || 0,
                        sex:      parseFloat(data.sex)      || 0,
                        cp:       parseFloat(data.cp)       || 0,
                        trestbps: parseFloat(data.trestbps) || 0,
                        chol:     parseFloat(data.chol)     || 0,
                        fbs:      parseFloat(data.fbs)      || 0,
                        restecg:  parseFloat(data.restecg)  || 0,
                        thalach:  parseFloat(data.thalach)  || 0,
                        exang:    parseFloat(data.exang)    || 0,
                        oldpeak:  parseFloat(data.oldpeak)  || 0,
                        slope:    parseFloat(data.slope)    || 0,
                        ca:       parseFloat(data.ca)       || 0,
                        thal:     parseFloat(data.thal)     || 0,
                    };

                    resultSection.classList.remove('hidden');
                    form.classList.remove('hidden');

                    setTimeout(() => {
                        probFill.style.width = probPercent + '%';
                        probText.innerText   = probPercent + '%';
                        if (typeof animateGauge === 'function') animateGauge(parseFloat(probPercent));

                        if (result.prediction === 1) {
                            resultTitle.innerHTML = '<span class="text-danger">⚠️ High Risk Detected</span>';
                            resultDesc.innerHTML  = 'Based on your vitals, our AI detects patterns commonly associated with Heart Disease. Please review the advice below and consult your doctor.';
                            probFill.style.background = 'linear-gradient(90deg, #ef4444, #b91c1c)';
                        } else {
                            resultTitle.innerHTML = '<span class="text-success">✅ Low Risk Assessed</span>';
                            resultDesc.innerHTML  = 'Great news! Your vitals fall within healthy margins. Keep up your healthy habits!';
                            probFill.style.background = 'linear-gradient(90deg, #10b981, #047857)';
                            if (typeof confetti !== 'undefined') {
                                confetti({ particleCount: 120, spread: 80, origin: { y: 0.6 }, colors: ['#10b981', '#3b82f6', '#86efac', '#60a5fa'] });
                            }
                        }

                        if (result.insights) {
                            insightsContainer.innerHTML = result.insights;
                            insightsContainer.classList.remove('hidden');
                        }
                        if (resultActions) resultActions.classList.remove('hidden');

                    }, 100);

                } else {
                    alert('Error in AI calculation: ' + result.error);
                    form.classList.remove('hidden');
                }
            }, 2800);

        } catch (error) {
            console.error('Fetch Error: ', error);
            if (typeof stopMessages === 'function') stopMessages();
            alert('Server connection failed.');
            ekgContainer.classList.add('hidden');
            form.classList.remove('hidden');
        }
    });
}
