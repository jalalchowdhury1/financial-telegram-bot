
describe('FrontRunner artifact digit', () => {
    const { frontRunnerText } = require('../sheetsCascade');
    test('strips the n8n digit after ")" but keeps real text', () => {
        expect(frontRunnerText('BIL (T-Bill ETF)1\n\n\nThis message was sent automatically with n8n')).toBe('BIL (T-Bill ETF)');
        expect(frontRunnerText('TQQQ (3x Nasdaq)')).toBe('TQQQ (3x Nasdaq)');
        expect(frontRunnerText('UVXY 2')).toBe('UVXY 2');
    });
});
