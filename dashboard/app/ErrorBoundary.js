'use client';

import React from 'react';

export default class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error('ErrorBoundary caught an error:', error, errorInfo);
        this.setState({ errorInfo });
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    padding: '16px',
                    borderRadius: '8px',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    backgroundColor: 'rgba(239, 68, 68, 0.05)',
                    color: 'rgba(255, 255, 255, 0.7)',
                    fontSize: '0.8rem',
                    textAlign: 'left',
                    fontFamily: 'monospace',
                    overflowX: 'auto'
                }}>
                    <div style={{ fontWeight: 600, color: '#ef4444', marginBottom: '8px', textAlign: 'center' }}>
                        ⚠️ Component Error: Data parsing failed.
                    </div>
                    <details style={{ cursor: 'pointer', opacity: 0.8 }}>
                        <summary style={{ outline: 'none' }}>View Detailed Error Log</summary>
                        <div style={{ marginTop: '8px', padding: '8px', background: 'rgba(0,0,0,0.2)', borderRadius: '4px', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                            <div style={{ color: '#ef4444' }}>{this.state.error && this.state.error.toString()}</div>
                            <br />
                            <div style={{ color: '#9ca3af' }}>{this.state.errorInfo && this.state.errorInfo.componentStack}</div>
                        </div>
                    </details>
                </div>
            );
        }

        return this.props.children;
    }
}
