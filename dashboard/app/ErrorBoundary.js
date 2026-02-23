'use client';

import React from 'react';

export default class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error('ErrorBoundary caught an error:', error, errorInfo);
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
                    textAlign: 'center',
                    fontFamily: 'monospace'
                }}>
                    ⚠️ Component Error: Data parsing failed.
                </div>
            );
        }

        return this.props.children;
    }
}
