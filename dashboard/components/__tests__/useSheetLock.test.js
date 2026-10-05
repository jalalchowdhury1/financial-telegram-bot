import { render } from '@testing-library/react';
import { useRef } from 'react';
import useSheetLock, { lockPage, unlockPage, useSheetFocus, SHEET_OPEN_CLASS } from '../useSheetLock';

function Sheet({ open }) {
    useSheetLock(open);
    return null;
}

const html = () => document.documentElement;
const touchMove = (target, touches = 1) => {
    const e = new Event('touchmove', { bubbles: true, cancelable: true });
    Object.defineProperty(e, 'touches', { value: Array.from({ length: touches }, () => ({ clientX: 0, clientY: 0 })) });
    target.dispatchEvent(e);
    return e;
};

let scrollTo;
beforeEach(() => {
    // jsdom has no layout: fake the page offset and record restores
    window.scrollY = 0;
    scrollTo = jest.fn((x, y) => { window.scrollY = y; });
    window.scrollTo = scrollTo;
});
afterEach(() => {
    // never leak a lock into the next test
    while (html().classList.contains(SHEET_OPEN_CLASS)) unlockPage();
});

test('open adds html.sheet-open and stops page scroll; close removes both', () => {
    const { rerender } = render(<Sheet open={false} />);
    expect(html().classList.contains(SHEET_OPEN_CLASS)).toBe(false);
    rerender(<Sheet open />);
    expect(html().classList.contains(SHEET_OPEN_CLASS)).toBe(true);
    expect(html().style.overflow).toBe('hidden');
    expect(document.body.style.overflow).toBe('hidden');
    rerender(<Sheet open={false} />);
    expect(html().classList.contains(SHEET_OPEN_CLASS)).toBe(false);
    expect(html().style.overflow).toBe('');
    expect(document.body.style.overflow).toBe('');
});

test('earlier inline overflow styles come back exactly', () => {
    document.body.style.overflow = 'clip';
    const { unmount } = render(<Sheet open />);
    expect(document.body.style.overflow).toBe('hidden');
    unmount();
    expect(document.body.style.overflow).toBe('clip');
    document.body.style.overflow = '';
});

test('closing puts the page back at the exact scrollY it had when the sheet opened', () => {
    window.scrollY = 1234;
    const { rerender } = render(<Sheet open />);
    window.scrollY = 0; // whatever moved it while open (an iOS bounce, a focus jump)
    rerender(<Sheet open={false} />);
    expect(scrollTo).toHaveBeenLastCalledWith(0, 1234);
    expect(window.scrollY).toBe(1234);
});

test('no needless scroll when the page never moved', () => {
    window.scrollY = 500;
    const { rerender } = render(<Sheet open />);
    rerender(<Sheet open={false} />);
    expect(scrollTo).not.toHaveBeenCalled();
});

test('nested sheets: the page unlocks only when the LAST one closes', () => {
    window.scrollY = 300;
    const a = render(<Sheet open />);
    const b = render(<Sheet open />);
    a.unmount();
    expect(html().classList.contains(SHEET_OPEN_CLASS)).toBe(true);
    expect(html().style.overflow).toBe('hidden');
    b.unmount();
    expect(html().classList.contains(SHEET_OPEN_CLASS)).toBe(false);
    expect(html().style.overflow).toBe('');
});

test('unmounting while open (a refresh resetting the card) cleans up', () => {
    const { unmount } = render(<Sheet open />);
    expect(html().classList.contains(SHEET_OPEN_CLASS)).toBe(true);
    unmount();
    expect(html().classList.contains(SHEET_OPEN_CLASS)).toBe(false);
    expect(html().style.overflow).toBe('');
});

test('an extra unlock never goes negative or unlocks a later sheet', () => {
    unlockPage(); unlockPage();
    const { unmount } = render(<Sheet open />);
    expect(html().classList.contains(SHEET_OPEN_CLASS)).toBe(true);
    unmount();
    expect(html().classList.contains(SHEET_OPEN_CLASS)).toBe(false);
});

test('while locked, a swipe outside the sheet is cancelled; the sheet itself still scrolls', () => {
    const sheet = document.createElement('div');
    sheet.setAttribute('data-sheet-scroll', '');
    Object.defineProperty(sheet, 'scrollHeight', { value: 900 });
    Object.defineProperty(sheet, 'clientHeight', { value: 400 });
    const inner = document.createElement('p');
    sheet.appendChild(inner);
    const backdrop = document.createElement('div');
    document.body.append(backdrop, sheet);

    expect(touchMove(backdrop).defaultPrevented).toBe(false); // not locked: untouched
    lockPage();
    expect(touchMove(backdrop).defaultPrevented).toBe(true);
    expect(touchMove(inner).defaultPrevented).toBe(false);
    expect(touchMove(backdrop, 2).defaultPrevented).toBe(false); // pinch-zoom stays
    unlockPage();
    expect(touchMove(backdrop).defaultPrevented).toBe(false);
    backdrop.remove(); sheet.remove();
});

test('a sheet whose content fits does not hand the swipe to the page either', () => {
    const sheet = document.createElement('div');
    sheet.setAttribute('data-sheet-scroll', '');
    Object.defineProperty(sheet, 'scrollHeight', { value: 300 });
    Object.defineProperty(sheet, 'clientHeight', { value: 300 });
    document.body.append(sheet);
    lockPage();
    expect(touchMove(sheet).defaultPrevented).toBe(true);
    unlockPage();
    sheet.remove();
});

describe('useSheetFocus — keyboard + screen-reader focus follows the sheet', () => {
    function FocusSheet({ open }) {
        const ref = useRef(null);
        useSheetFocus(open, ref);
        if (!open) return null;
        return (
            <div role="dialog" ref={ref}>
                <button className="sheet-x">×</button>
                <a href="#x">link</a>
            </div>
        );
    }
    function Page({ open }) {
        return (<><button data-testid="trigger">open</button><FocusSheet open={open} /></>);
    }
    const tab = (shift = false) => {
        const e = new KeyboardEvent('keydown', { key: 'Tab', shiftKey: shift, bubbles: true, cancelable: true });
        document.activeElement.dispatchEvent(e);
        return e;
    };

    test('open moves focus to the ×; close gives it back to the trigger', () => {
        const { rerender, getByTestId } = render(<Page open={false} />);
        getByTestId('trigger').focus();
        rerender(<Page open />);
        expect(document.activeElement.className).toBe('sheet-x');
        rerender(<Page open={false} />);
        expect(document.activeElement).toBe(getByTestId('trigger'));
    });

    test('Tab wraps inside the sheet, never onto the locked page behind it', () => {
        const { rerender } = render(<Page open={false} />);
        rerender(<Page open />);
        const x = document.querySelector('.sheet-x');
        const link = document.querySelector('a');
        link.focus();
        expect(tab().defaultPrevented).toBe(true);
        expect(document.activeElement).toBe(x);
        expect(tab(true).defaultPrevented).toBe(true);
        expect(document.activeElement).toBe(link);
        // a Tab in the middle is left to the browser
        x.focus();
        expect(tab().defaultPrevented).toBe(false);
    });

    test('a trigger that is gone by close time is skipped safely', () => {
        function Gone({ open, showTrigger }) {
            return (<>{showTrigger && <button data-testid="t">t</button>}<FocusSheet open={open} /></>);
        }
        const { rerender, getByTestId } = render(<Gone open={false} showTrigger />);
        getByTestId('t').focus();
        rerender(<Gone open showTrigger={false} />);
        expect(() => rerender(<Gone open={false} showTrigger={false} />)).not.toThrow();
    });
});
