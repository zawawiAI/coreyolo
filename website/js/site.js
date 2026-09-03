(function () {
  const toggle = document.querySelector(".nav-toggle");
  const links = document.querySelector(".nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", function () {
      const open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", async function () {
      const sel = btn.getAttribute("data-copy");
      const el = document.querySelector(sel);
      if (!el) return;
      const text = el.innerText.trim();
      let ok = false;
      try {
        await navigator.clipboard.writeText(text);
        ok = true;
      } catch (err) {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand("copy");
        ta.remove();
      }
      const prev = btn.textContent;
      btn.textContent = ok ? "Copied" : "Copy failed";
      setTimeout(function () {
        btn.textContent = prev;
      }, 1400);
    });
  });

  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll("[data-panel]");
  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      const id = tab.getAttribute("data-tab");
      tabs.forEach(function (t) {
        t.setAttribute("aria-selected", t === tab ? "true" : "false");
      });
      panels.forEach(function (p) {
        p.hidden = p.getAttribute("data-panel") !== id;
      });
    });
  });
})();
