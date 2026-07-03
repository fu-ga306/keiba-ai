/* 競馬AI予想 — メインJS */

document.addEventListener('DOMContentLoaded', () => {
  animateRings();
  initRefreshLink();
});

// 荒れスコアリングのアニメーション（ページロード時）
function animateRings() {
  const bars = document.querySelectorAll('.ring-bar');
  bars.forEach(bar => {
    const original = bar.getAttribute('stroke-dasharray');
    bar.setAttribute('stroke-dasharray', '0 100');
    requestAnimationFrame(() => {
      setTimeout(() => {
        bar.style.transition = 'stroke-dasharray 0.8s ease-out';
        bar.setAttribute('stroke-dasharray', original);
      }, 100);
    });
  });
}

// データ更新リンク
function initRefreshLink() {
  const refreshLinks = document.querySelectorAll('.refresh-link');
  refreshLinks.forEach(link => {
    link.addEventListener('click', async (e) => {
      e.preventDefault();
      link.textContent = '🔄 更新中...';
      try {
        await fetch('/api/refresh');
        link.textContent = '✅ 更新完了';
        setTimeout(() => location.reload(), 800);
      } catch {
        link.textContent = '❌ 更新失敗';
      }
    });
  });
}
