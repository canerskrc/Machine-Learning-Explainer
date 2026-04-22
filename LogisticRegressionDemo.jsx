import { useEffect, useMemo, useRef, useState } from "react";

const COLORS = {
  bg: "#f5f7fb",
  surface: "#ffffff",
  surfaceSoft: "#eef2f9",
  ink: "#18212f",
  inkSoft: "#526075",
  inkMute: "#8592a6",
  rule: "#dde4ef",
  grid: "#e9eef6",
  accent: "#4f46e5",
  accentSoft: "#e1e7ff",
  class0: "#0f172a",
  class1: "#14b8a6",
  class0Soft: "#d7deea",
  class1Soft: "#d8f6f1",
  boundary: "#ef4444",
};

const FONT_DISPLAY = "'Instrument Serif', Georgia, serif";
const FONT_BODY = "'Inter', -apple-system, sans-serif";
const FONT_MONO = "'JetBrains Mono', 'SF Mono', monospace";

const INITIAL_POINTS = [
  { x: 1.2, y: 0.08, label: 0 },
  { x: 2.0, y: 0.12, label: 0 },
  { x: 2.6, y: 0.18, label: 0 },
  { x: 3.5, y: 0.22, label: 0 },
  { x: 4.1, y: 0.35, label: 0 },
  { x: 5.0, y: 0.62, label: 1 },
  { x: 5.8, y: 0.74, label: 1 },
  { x: 6.5, y: 0.81, label: 1 },
  { x: 7.4, y: 0.88, label: 1 },
  { x: 8.2, y: 0.93, label: 1 },
];

function sigmoid(z) {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

function trainLogisticRegression(points, learningRate = 0.18, steps = 500, lambda = 0) {
  let w = 0;
  let b = 0;
  const m = points.length || 1;

  for (let step = 0; step < steps; step++) {
    let dw = 0;
    let db = 0;

    for (const p of points) {
      const pred = sigmoid(w * p.x + b);
      const error = pred - p.label;
      dw += error * p.x;
      db += error;
    }

    dw = dw / m + lambda * w;
    db = db / m;

    w -= learningRate * dw;
    b -= learningRate * db;
  }

  return { w, b };
}

function computeLogLoss(points, w, b, lambda = 0) {
  if (!points.length) return 0;
  let sum = 0;
  for (const p of points) {
    const pred = Math.min(0.999999, Math.max(0.000001, sigmoid(w * p.x + b)));
    sum += -(p.label * Math.log(pred) + (1 - p.label) * Math.log(1 - pred));
  }
  return sum / points.length + 0.5 * lambda * w * w;
}

function computeAccuracy(points, w, b, threshold = 0.5) {
  if (!points.length) return 0;
  let ok = 0;
  for (const p of points) {
    const pred = sigmoid(w * p.x + b) >= threshold ? 1 : 0;
    if (pred === p.label) ok += 1;
  }
  return ok / points.length;
}

export default function LogisticRegressionDemo() {
  const [points, setPoints] = useState(INITIAL_POINTS);
  const [layer, setLayer] = useState("intuition");
  const [draggingIdx, setDraggingIdx] = useState(null);
  const [hoverPoint, setHoverPoint] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [lambda, setLambda] = useState(0.02);
  const [learningRate, setLearningRate] = useState(0.18);
  const [showProbabilities, setShowProbabilities] = useState(true);
  const [addClass, setAddClass] = useState(1);
  const svgRef = useRef(null);

  const W = 640;
  const H = 420;
  const PAD = { top: 28, right: 28, bottom: 52, left: 56 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const xScale = (x) => PAD.left + (x / 10) * plotW;
  const yScale = (y) => PAD.top + plotH - y * plotH;
  const xInv = (px) => ((px - PAD.left) / plotW) * 10;
  const yInv = (py) => Math.max(0, Math.min(1, (PAD.top + plotH - py) / plotH));

  const { w, b } = useMemo(
    () => trainLogisticRegression(points, learningRate, 550, lambda),
    [points, learningRate, lambda]
  );

  const logLoss = useMemo(() => computeLogLoss(points, w, b, lambda), [points, w, b, lambda]);
  const accuracy = useMemo(() => computeAccuracy(points, w, b, threshold), [points, w, b, threshold]);
  const boundaryX = Math.abs(w) < 1e-8 ? null : -b / w;

  const curvePoints = useMemo(() => {
    const arr = [];
    for (let i = 0; i <= 120; i++) {
      const x = (i / 120) * 10;
      arr.push({ x, y: sigmoid(w * x + b) });
    }
    return arr;
  }, [w, b]);

  const curvePath = curvePoints
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.x)} ${yScale(p.y)}`)
    .join(" ");

  function handleSvgClick(e) {
    if (draggingIdx !== null) return;
    const rect = svgRef.current.getBoundingClientRect();
    const scale = W / rect.width;
    const px = (e.clientX - rect.left) * scale;
    const py = (e.clientY - rect.top) * scale;

    if (px < PAD.left || px > W - PAD.right || py < PAD.top || py > H - PAD.bottom) return;

    for (let i = 0; i < points.length; i++) {
      const dx = xScale(points[i].x) - px;
      const dy = yScale(points[i].y) - py;
      if (dx * dx + dy * dy < 140) return;
    }

    setPoints([...points, { x: xInv(px), y: yInv(py), label: addClass }]);
  }

  function handlePointDown(e, idx) {
    e.stopPropagation();
    setDraggingIdx(idx);
  }

  function handleMove(e) {
    if (draggingIdx === null) return;
    const rect = svgRef.current.getBoundingClientRect();
    const scale = W / rect.width;
    const px = Math.max(PAD.left, Math.min(W - PAD.right, (e.clientX - rect.left) * scale));
    const py = Math.max(PAD.top, Math.min(H - PAD.bottom, (e.clientY - rect.top) * scale));

    setPoints((prev) => {
      const next = [...prev];
      next[draggingIdx] = {
        ...next[draggingIdx],
        x: xInv(px),
        y: yInv(py),
      };
      return next;
    });
  }

  function handleUp() {
    setDraggingIdx(null);
  }

  function handlePointRightClick(e, idx) {
    e.preventDefault();
    setPoints((prev) => prev.filter((_, i) => i !== idx));
  }

  function flipLabels() {
    setPoints((prev) => prev.map((p) => ({ ...p, label: p.label === 1 ? 0 : 1 })));
  }

  function addNoise() {
    setPoints((prev) =>
      prev.map((p) => ({
        ...p,
        x: Math.max(0.4, Math.min(9.6, p.x + (Math.random() - 0.5) * 0.8)),
        y: Math.max(0.04, Math.min(0.96, p.y + (Math.random() - 0.5) * 0.1)),
      }))
    );
  }

  useEffect(() => {
    if (draggingIdx !== null) {
      window.addEventListener("mousemove", handleMove);
      window.addEventListener("mouseup", handleUp);
      return () => {
        window.removeEventListener("mousemove", handleMove);
        window.removeEventListener("mouseup", handleUp);
      };
    }
  }, [draggingIdx]);

  const equation = `p(y=1|x) = 1 / (1 + e^-(${w.toFixed(2)}x ${b >= 0 ? "+" : "-"} ${Math.abs(b).toFixed(2)}))`;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        .log-root { background: ${COLORS.bg}; color: ${COLORS.ink}; font-family: ${FONT_BODY}; padding: 48px 56px; min-height: 100vh; }
        .log-root * { box-sizing: border-box; }
        .log-eyebrow { font-family: ${FONT_MONO}; font-size: 11px; letter-spacing: 0.12em; color: ${COLORS.inkMute}; text-transform: uppercase; margin-bottom: 12px; }
        .log-title { font-family: ${FONT_DISPLAY}; font-size: 44px; line-height: 1.04; font-weight: 400; letter-spacing: -0.03em; margin: 0 0 16px; }
        .log-title em { color: ${COLORS.accent}; font-style: italic; }
        .log-lede { max-width: 680px; color: ${COLORS.inkSoft}; line-height: 1.65; font-size: 16px; margin: 0 0 36px; }

        .log-tabs { display: flex; gap: 28px; border-bottom: 1px solid ${COLORS.rule}; margin-bottom: 30px; }
        .log-tab { background: none; border: none; padding: 0 0 14px; cursor: pointer; font-size: 13px; color: ${COLORS.inkMute}; position: relative; font-weight: 500; }
        .log-tab.active { color: ${COLORS.ink}; }
        .log-tab.active::after { content: ''; position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; background: ${COLORS.accent}; }
        .log-tab-num { font-family: ${FONT_MONO}; font-size: 10px; margin-right: 8px; color: ${COLORS.inkMute}; }

        .log-intro { font-family: ${FONT_DISPLAY}; font-size: 19px; line-height: 1.5; color: ${COLORS.inkSoft}; margin-bottom: 28px; max-width: 620px; }

        .log-grid { display: grid; grid-template-columns: 1fr 330px; gap: 42px; align-items: start; }
        .log-canvas { background: ${COLORS.surface}; border: 1px solid ${COLORS.rule}; border-radius: 18px; overflow: hidden; box-shadow: 0 10px 30px rgba(24,33,47,0.05); }
        .log-svg { display: block; width: 100%; height: auto; cursor: crosshair; user-select: none; }
        .log-canvas-hint { display: flex; justify-content: space-between; gap: 16px; padding: 14px 18px; border-top: 1px solid ${COLORS.rule}; color: ${COLORS.inkMute}; font-family: ${FONT_MONO}; font-size: 11px; }

        .log-equation-box { margin-top: 18px; background: ${COLORS.surface}; border: 1px solid ${COLORS.rule}; border-radius: 16px; padding: 18px 20px; }
        .log-equation-label { font-family: ${FONT_MONO}; font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: ${COLORS.inkMute}; margin-bottom: 10px; }
        .log-equation { font-family: ${FONT_MONO}; font-size: 14px; color: ${COLORS.ink}; line-height: 1.7; word-break: break-word; }

        .log-panel { background: transparent; }
        .log-panel-title { font-family: ${FONT_DISPLAY}; font-size: 22px; font-weight: 400; margin: 0 0 18px; }
        .log-row { display: flex; justify-content: space-between; align-items: baseline; padding: 11px 0; border-bottom: 1px dashed ${COLORS.rule}; }
        .log-row-label { font-size: 13px; color: ${COLORS.inkSoft}; }
        .log-row-value { font-family: ${FONT_MONO}; font-size: 14px; color: ${COLORS.ink}; font-weight: 500; }
        .log-row-value.accent { color: ${COLORS.accent}; }

        .log-control { margin-top: 20px; }
        .log-control-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }
        .log-control-label { font-size: 13px; font-weight: 600; color: ${COLORS.ink}; }
        .log-control-value { font-family: ${FONT_MONO}; font-size: 13px; color: ${COLORS.accent}; }
        .log-slider { width: 100%; appearance: none; -webkit-appearance: none; height: 4px; background: ${COLORS.rule}; border-radius: 999px; outline: none; }
        .log-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: ${COLORS.accent}; border: 2px solid white; box-shadow: 0 0 0 1px ${COLORS.accent}; cursor: pointer; }
        .log-slider::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: ${COLORS.accent}; border: none; cursor: pointer; }
        .log-control-hint { margin-top: 8px; font-size: 11px; color: ${COLORS.inkMute}; line-height: 1.5; }

        .log-toggle { display: flex; align-items: center; gap: 10px; margin-top: 18px; font-size: 13px; color: ${COLORS.ink}; }
        .log-segment { display: flex; gap: 8px; margin-top: 18px; }
        .log-chip { border: 1px solid ${COLORS.rule}; background: ${COLORS.surface}; color: ${COLORS.inkSoft}; padding: 9px 12px; border-radius: 999px; font-size: 12px; cursor: pointer; transition: 0.15s ease; }
        .log-chip.active.class0 { background: ${COLORS.class0Soft}; color: ${COLORS.class0}; border-color: ${COLORS.class0Soft}; }
        .log-chip.active.class1 { background: ${COLORS.class1Soft}; color: ${COLORS.class1}; border-color: ${COLORS.class1Soft}; }

        .log-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 28px; }
        .log-btn { border: 1px solid ${COLORS.rule}; background: ${COLORS.surface}; color: ${COLORS.ink}; border-radius: 999px; padding: 10px 14px; font-size: 12px; font-weight: 500; cursor: pointer; transition: 0.15s ease; }
        .log-btn:hover { border-color: ${COLORS.accent}; color: ${COLORS.accent}; }

        .log-note { margin-top: 30px; background: ${COLORS.surface}; border: 1px solid ${COLORS.rule}; border-radius: 16px; padding: 20px; }
        .log-note-label { font-family: ${FONT_MONO}; font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: ${COLORS.accent}; margin-bottom: 10px; }
        .log-note-text { margin: 0; color: ${COLORS.inkSoft}; line-height: 1.65; font-size: 14px; }
        .log-note-text strong { color: ${COLORS.ink}; }

        .log-code-label { font-family: ${FONT_MONO}; font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: ${COLORS.inkMute}; margin: 30px 0 10px; }
        .log-code { background: #111827; color: #eef2ff; border-radius: 16px; padding: 18px 20px; font-family: ${FONT_MONO}; font-size: 12px; line-height: 1.7; overflow-x: auto; }
        .log-code .kw { color: #a5b4fc; }
        .log-code .fn { color: #5eead4; }
        .log-code .cm { color: #94a3b8; }

        @media (max-width: 860px) {
          .log-root { padding: 28px 20px; }
          .log-title { font-size: 34px; }
          .log-grid { grid-template-columns: 1fr; }
        }
      `}</style>

      <div className="log-root">
        <div className="log-eyebrow">Demo 02 · Supervised learning · Classification</div>
        <h1 className="log-title">
          Logistic regression, <em>seen as probability</em>.
        </h1>
        <p className="log-lede">
          Bu demo artık bir çizgi uydurmuyor; bir noktanın Class 1 olma olasılığını hesaplıyor.
          Noktaları sürükle, yeni örnekler ekle, threshold değiştir ve sigmoid eğrisinin karar
          mantığını canlı olarak izle.
        </p>

        <div className="log-tabs">
          {[
            { id: "intuition", label: "Intuition", num: "01" },
            { id: "mechanics", label: "Mechanics", num: "02" },
            { id: "production", label: "Production", num: "03" },
          ].map((tab) => (
            <button
              key={tab.id}
              className={`log-tab ${layer === tab.id ? "active" : ""}`}
              onClick={() => setLayer(tab.id)}
            >
              <span className="log-tab-num">{tab.num}</span>
              {tab.label}
            </button>
          ))}
        </div>

        {layer === "intuition" && (
          <p className="log-intro">
            Logistic regression “hangi sınıfa daha yakın?” diye düşünmez; “bu örneğin sınıf 1 olma
            olasılığı nedir?” diye sorar. Sonuç doğrusal bir skorla başlar, sonra sigmoid fonksiyonu
            ile 0 ile 1 arasına sıkıştırılır.
          </p>
        )}

        {layer === "mechanics" && (
          <p className="log-intro">
            Amaç squared error değil, <strong>log loss</strong> minimizasyonudur. Model yanlış ve
            aşırı emin tahminler verdiğinde sert biçimde cezalandırılır. Bu yüzden logistic regression,
            classification problemlerinde lineer regresyondan daha anlamlıdır.
          </p>
        )}

        {layer === "production" && (
          <p className="log-intro">
            Gerçek sistemlerde sadece accuracy yetmez. Threshold ayarı, class imbalance, calibration,
            false positive maliyeti ve regularization birlikte değerlendirilir. Production kararı,
            yalnızca matematiksel değil operasyonel bir karardır.
          </p>
        )}

        <div className="log-grid">
          <div>
            <div className="log-canvas">
              <svg ref={svgRef} className="log-svg" viewBox={`0 0 ${W} ${H}`} onClick={handleSvgClick}>
                <defs>
                  <pattern id="gridLog" width="32" height="32" patternUnits="userSpaceOnUse">
                    <path d="M 32 0 L 0 0 0 32" fill="none" stroke={COLORS.grid} strokeWidth="1" />
                  </pattern>
                </defs>

                <rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} fill="url(#gridLog)" />

                {[0, 2, 4, 6, 8, 10].map((v) => (
                  <g key={`x-${v}`}>
                    <line x1={xScale(v)} y1={PAD.top + plotH} x2={xScale(v)} y2={PAD.top + plotH + 6} stroke={COLORS.inkMute} strokeWidth="1" />
                    <text x={xScale(v)} y={PAD.top + plotH + 21} textAnchor="middle" fontSize="11" fill={COLORS.inkMute} fontFamily={FONT_MONO}>
                      {v}
                    </text>
                  </g>
                ))}

                {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                  <g key={`y-${v}`}>
                    <line x1={PAD.left - 6} y1={yScale(v)} x2={PAD.left} y2={yScale(v)} stroke={COLORS.inkMute} strokeWidth="1" />
                    <text x={PAD.left - 12} y={yScale(v) + 4} textAnchor="end" fontSize="11" fill={COLORS.inkMute} fontFamily={FONT_MONO}>
                      {v.toFixed(v === 0 || v === 1 ? 0 : 2)}
                    </text>
                  </g>
                ))}

                <line x1={PAD.left} y1={PAD.top + plotH} x2={W - PAD.right} y2={PAD.top + plotH} stroke={COLORS.ink} strokeWidth="1" />
                <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH} stroke={COLORS.ink} strokeWidth="1" />

                <text x={W - PAD.right} y={PAD.top + plotH + 40} textAnchor="end" fontSize="11" fill={COLORS.inkSoft} fontFamily={FONT_MONO}>
                  x — feature value
                </text>
                <text x={PAD.left - 8} y={PAD.top - 12} textAnchor="start" fontSize="11" fill={COLORS.inkSoft} fontFamily={FONT_MONO}>
                  probability
                </text>

                {showProbabilities && (
                  <line
                    x1={PAD.left}
                    y1={yScale(threshold)}
                    x2={W - PAD.right}
                    y2={yScale(threshold)}
                    stroke={COLORS.boundary}
                    strokeWidth="1.5"
                    strokeDasharray="6 6"
                    opacity="0.7"
                  />
                )}

                <path d={curvePath} fill="none" stroke={COLORS.accent} strokeWidth="3" />

                {boundaryX !== null && boundaryX >= 0 && boundaryX <= 10 && (
                  <g>
                    <line
                      x1={xScale(boundaryX)}
                      y1={PAD.top}
                      x2={xScale(boundaryX)}
                      y2={PAD.top + plotH}
                      stroke={COLORS.boundary}
                      strokeWidth="2"
                      strokeDasharray="5 5"
                    />
                    <text
                      x={xScale(boundaryX)}
                      y={PAD.top + 18}
                      textAnchor="middle"
                      fontSize="10"
                      fill={COLORS.boundary}
                      fontFamily={FONT_MONO}
                    >
                      decision boundary
                    </text>
                  </g>
                )}

                {points.map((p, i) => {
                  const pred = sigmoid(w * p.x + b);
                  const isPositive = p.label === 1;
                  return (
                    <g key={i}>
                      {showProbabilities && (
                        <line
                          x1={xScale(p.x)}
                          y1={yScale(p.y)}
                          x2={xScale(p.x)}
                          y2={yScale(pred)}
                          stroke={COLORS.inkMute}
                          strokeDasharray="3 4"
                          opacity="0.35"
                        />
                      )}
                      <circle
                        cx={xScale(p.x)}
                        cy={yScale(p.y)}
                        r={hoverPoint === i || draggingIdx === i ? 8 : 6}
                        fill={isPositive ? COLORS.class1 : COLORS.class0}
                        stroke="#ffffff"
                        strokeWidth="2"
                        style={{ cursor: "move" }}
                        onMouseDown={(e) => handlePointDown(e, i)}
                        onMouseEnter={() => setHoverPoint(i)}
                        onMouseLeave={() => setHoverPoint(null)}
                        onContextMenu={(e) => handlePointRightClick(e, i)}
                      />
                      {hoverPoint === i && (
                        <text
                          x={xScale(p.x) + 10}
                          y={yScale(p.y) - 10}
                          fontFamily={FONT_MONO}
                          fontSize="10"
                          fill={COLORS.inkSoft}
                        >
                          p={pred.toFixed(2)} · y={p.label}
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
              <div className="log-canvas-hint">
                <span>Tıkla → yeni örnek · Sürükle → taşı · Sağ tık → sil</span>
                <span>{points.length} samples</span>
              </div>
            </div>

            <div className="log-equation-box">
              <div className="log-equation-label">Model form</div>
              <div className="log-equation">{equation}</div>
            </div>

            {layer === "mechanics" && (
              <>
                <div className="log-code-label">Minimal training logic</div>
                <div className="log-code">
                  <span className="cm"># gradient descent with log loss</span>
                  {"\n"}
                  <span className="kw">for</span> point <span className="kw">in</span> data:{"\n"}
                  &nbsp;&nbsp;pred = <span className="fn">sigmoid</span>(w * x + b){"\n"}
                  &nbsp;&nbsp;error = pred - y{"\n"}
                  &nbsp;&nbsp;dw += error * x{"\n"}
                  &nbsp;&nbsp;db += error{"\n\n"}
                  w -= lr * (dw / m + lambda * w){"\n"}
                  b -= lr * (db / m)
                </div>
              </>
            )}
          </div>

          <div className="log-panel">
            <h3 className="log-panel-title">Live metrics</h3>

            <div className="log-row">
              <span className="log-row-label">Log loss</span>
              <span className="log-row-value accent">{logLoss.toFixed(4)}</span>
            </div>
            <div className="log-row">
              <span className="log-row-label">Accuracy</span>
              <span className="log-row-value">{(accuracy * 100).toFixed(1)}%</span>
            </div>
            <div className="log-row">
              <span className="log-row-label">Weight (w)</span>
              <span className="log-row-value">{w.toFixed(3)}</span>
            </div>
            <div className="log-row">
              <span className="log-row-label">Bias (b)</span>
              <span className="log-row-value">{b.toFixed(3)}</span>
            </div>
            <div className="log-row">
              <span className="log-row-label">Boundary x</span>
              <span className="log-row-value">{boundaryX === null ? "—" : boundaryX.toFixed(2)}</span>
            </div>

            <div className="log-control">
              <div className="log-control-header">
                <span className="log-control-label">Decision threshold</span>
                <span className="log-control-value">{threshold.toFixed(2)}</span>
              </div>
              <input
                className="log-slider"
                type="range"
                min="0.1"
                max="0.9"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
              />
              <div className="log-control-hint">
                Threshold yükseldikçe model daha seçici davranır. Özellikle false positive maliyeti yüksekse önemlidir.
              </div>
            </div>

            <div className="log-control">
              <div className="log-control-header">
                <span className="log-control-label">Regularization λ</span>
                <span className="log-control-value">{lambda.toFixed(2)}</span>
              </div>
              <input
                className="log-slider"
                type="range"
                min="0"
                max="0.5"
                step="0.01"
                value={lambda}
                onChange={(e) => setLambda(parseFloat(e.target.value))}
              />
              <div className="log-control-hint">
                Büyük λ, weight değerini baskılar. Bu da daha stabil ama daha muhafazakâr karar sınırı üretir.
              </div>
            </div>

            <div className="log-control">
              <div className="log-control-header">
                <span className="log-control-label">Learning rate</span>
                <span className="log-control-value">{learningRate.toFixed(2)}</span>
              </div>
              <input
                className="log-slider"
                type="range"
                min="0.05"
                max="0.40"
                step="0.01"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              />
              <div className="log-control-hint">
                Çok düşük olursa öğrenme yavaşlar, çok yüksek olursa salınım artar. Demo için orta bant ideal.
              </div>
            </div>

            <label className="log-toggle">
              <input type="checkbox" checked={showProbabilities} onChange={(e) => setShowProbabilities(e.target.checked)} />
              Probability guide lines göster
            </label>

            <div className="log-segment">
              <button
                className={`log-chip ${addClass === 0 ? "active class0" : ""}`}
                onClick={() => setAddClass(0)}
              >
                Add Class 0
              </button>
              <button
                className={`log-chip ${addClass === 1 ? "active class1" : ""}`}
                onClick={() => setAddClass(1)}
              >
                Add Class 1
              </button>
            </div>

            <div className="log-actions">
              <button className="log-btn" onClick={() => setPoints(INITIAL_POINTS)}>Reset</button>
              <button className="log-btn" onClick={addNoise}>Add noise</button>
              <button className="log-btn" onClick={flipLabels}>Flip labels</button>
            </div>

            <div className="log-note">
              <div className="log-note-label">Reading the demo</div>
              <p className="log-note-text">
                <strong>Siyah noktalar</strong> Class 0, <strong>turkuaz noktalar</strong> Class 1.
                Mor sigmoid eğrisi her x değeri için <strong>Class 1 olasılığını</strong> verir.
                Kırmızı kesikli dikey çizgi ise modelin sınıf değiştirdiği yaklaşık karar sınırını gösterir.
              </p>
            </div>

            {layer === "production" && (
              <div className="log-note" style={{ marginTop: 16 }}>
                <div className="log-note-label">Production note</div>
                <p className="log-note-text">
                  Fraud detection, churn prediction veya medical triage gibi alanlarda threshold çoğu zaman 0.50 kalmaz.
                  Çünkü iş problemi “en doğru tahmin” değil, <strong>en doğru maliyet dengesi</strong> problemidir.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
