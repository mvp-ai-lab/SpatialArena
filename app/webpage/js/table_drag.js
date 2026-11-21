// Copyright (c) 2026 Yu Chen. All rights reserved.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

// Wide tables: drag with the mouse to scroll them sideways, with the method column held in place.
//
// Every `.table-container` that scrolls horizontally can be dragged left / right with the mouse (touch keeps its native
// panning); a drag never triggers the click (column sort) it started on. The columns up to and including the method
// column (`td.benchmark-method-name`, e.g. Rank + Method in the leaderboard; the first column when a table has none)
// stay fixed while the rest scrolls under them. Tables are picked up as they are rendered and re-measured on resize.
(() => {
    const DRAG_THRESHOLD_PX = 5;
    const INTERACTIVE = "a, input, select, textarea, label, [contenteditable]";

    // Number of leading columns to hold: through the method column, else the first one.
    function stickyColumnCount(table) {
        const row = [...(table.tBodies[0]?.rows || [])].find(tr => tr.cells.length > 1);
        if (!row) return 0;
        let column = 0;
        for (const cell of row.cells) {
            column += cell.colSpan || 1;
            if (cell.classList.contains("benchmark-method-name")) return column;
        }
        return 1;
    }

    // Each row's cells with their starting column, colspans counted.
    function cellsWithColumns(row) {
        let column = 0;
        return [...row.cells].map(cell => {
            const entry = { cell, start: column, span: cell.colSpan || 1 };
            column += entry.span;
            return entry;
        });
    }

    function layoutStickyColumns(table) {
        const count = stickyColumnCount(table);
        table.querySelectorAll(".sticky-col").forEach(cell => {
            cell.classList.remove("sticky-col", "sticky-col-last");
            cell.style.left = "";
        });
        if (!count) return;
        // Column offsets from the first body row whose leading cells are one column each.
        const reference = [...table.tBodies[0].rows].find(tr => cellsWithColumns(tr)
            .slice(0, count).every(entry => entry.span === 1)) || table.tBodies[0].rows[0];
        const lefts = [];
        let left = 0;
        for (const entry of cellsWithColumns(reference).slice(0, count)) {
            lefts[entry.start] = left;
            left += entry.cell.getBoundingClientRect().width;
        }
        for (const row of table.rows) {
            for (const entry of cellsWithColumns(row)) {
                if (entry.start + entry.span > count) break;
                entry.cell.classList.add("sticky-col");
                entry.cell.style.left = `${lefts[entry.start] ?? 0}px`;
                if (entry.start + entry.span === count) entry.cell.classList.add("sticky-col-last");
            }
        }
    }

    function updateScrollState(container) {
        const scrollable = container.scrollWidth > container.clientWidth + 1;
        container.classList.toggle("is-draggable", scrollable);
        container.classList.toggle("is-scrolled", scrollable && container.scrollLeft > 0);
    }

    function enableDrag(container) {
        let start = null;
        let dragged = false;
        container.addEventListener("pointerdown", event => {
            if (event.pointerType !== "mouse" || event.button !== 0) return;
            if (!container.classList.contains("is-draggable") || event.target.closest(INTERACTIVE)) return;
            start = { x: event.clientX, scrollLeft: container.scrollLeft, id: event.pointerId };
            dragged = false;
        });
        container.addEventListener("pointermove", event => {
            if (!start || event.pointerId !== start.id) return;
            const dx = event.clientX - start.x;
            if (!dragged && Math.abs(dx) < DRAG_THRESHOLD_PX) return;
            if (!dragged) {
                dragged = true;
                container.setPointerCapture(event.pointerId);
                container.classList.add("is-dragging");
                window.getSelection()?.removeAllRanges();
            }
            container.scrollLeft = start.scrollLeft - dx;
            event.preventDefault();
        });
        const end = event => {
            if (!start || event.pointerId !== start.id) return;
            start = null;
            container.classList.remove("is-dragging");
            if (container.hasPointerCapture(event.pointerId)) container.releasePointerCapture(event.pointerId);
        };
        container.addEventListener("pointerup", end);
        container.addEventListener("pointercancel", end);
        // The click that ends a drag must not sort a column or follow a link.
        container.addEventListener("click", event => {
            if (!dragged) return;
            dragged = false;
            event.preventDefault();
            event.stopPropagation();
        }, true);
        container.addEventListener("scroll", () => updateScrollState(container), { passive: true });
    }

    const resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
            const container = entry.target;
            container.querySelectorAll(":scope > table").forEach(layoutStickyColumns);
            updateScrollState(container);
        }
    });

    function setUp(container) {
        if (!container.dataset.tableDrag) {
            container.dataset.tableDrag = "1";
            enableDrag(container);
            resizeObserver.observe(container);
            const table = container.querySelector(":scope > table");
            if (table) resizeObserver.observe(table);
        }
        container.querySelectorAll(":scope > table").forEach(layoutStickyColumns);
        updateScrollState(container);
    }

    let pending = false;
    function scan() {
        pending = false;
        document.querySelectorAll(".table-container").forEach(setUp);
    }
    function schedule() {
        if (pending) return;
        pending = true;
        requestAnimationFrame(scan);
    }
    // Tables are (re)rendered by innerHTML on every tab or sort change: lay them out again after each (and only then,
    // not on every other DOM change of the page).
    const touchesTable = node => node.nodeType === 1
        && (node.closest(".table-container") || node.querySelector(".table-container"));
    new MutationObserver(records => {
        if (records.some(record => touchesTable(record.target) || [...record.addedNodes].some(touchesTable))) schedule();
    }).observe(document.documentElement, { childList: true, subtree: true });
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", schedule);
    else schedule();
})();
