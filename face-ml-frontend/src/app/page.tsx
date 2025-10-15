"use client"
import { useEffect, useRef, useState } from "react"

const BASE_URL = "/api/images";

export default function LabelingPage() {
  const [files, setFiles] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [scale, setScale] = useState({x: 1, y: 1});

  async function fetchFiles(){
    const res =  await fetch(`${BASE_URL}/list-files`);
    const data = await res.json();
    setFiles(data);
  }

  async function inferImage(filePath: string) {
    setLoading(true);
    const res = await fetch(`${BASE_URL}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_path: filePath }),
    });

    const data = await res.json();
    setLoading(false);

    const annotations = (data.annotations || []).map((a: any, i: number) => ({
      ...a,
      id: i + 1,
      active: true,
    }));

    setSelected({
      filePath,
      fileName: filePath.split("/").pop(),
      annotations,
    });

    console.log(`Source: ${data.source}`);
  }


  async function saveImage() {
    if (!selected) return;

    const valid =  selected.annotations.filter((a: any) => a.active).map((a: any) => ({
      label: a.label.trim(),
      bbox: a.bbox.map(Number),
      confidence: Number(a.confidence)
    }));

    await fetch(`${BASE_URL}/save`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        fileName: selected.fileName,
        filePath: selected.filePath,
        annotations: valid,
      }),
    });

    setFiles(prev => prev.map(f => 
      f.filePath === selected.filePath ? { ...f, hasBox: true } : f
    ));

    alert("Ảnh đã được lưu vào database!");
  }

  function prevImage(){
    if (currentIndex > 0){
      const newIndex =  currentIndex - 1;
      setCurrentIndex(newIndex);
      inferImage(files[newIndex].filePath);
    }
  }

  function nextImage(){
    if (currentIndex < files.length - 1){
      const newIndex = currentIndex + 1;
      setCurrentIndex(newIndex);
      inferImage(files[newIndex].filePath);
    }
  }

  function updateLabel(index: number, newLabel: string){
    const anns = [...selected.annotations];
    anns[index].label = newLabel;
    setSelected({...selected, annotations: anns});
  }

  function toggleActive(index: number){
    const anns = [...selected.annotations];
    anns[index].active = !anns[index].active;
    setSelected({...selected, annotations: anns});
  }

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;

    const updateScale = () => {
      setScale({
        x: img.clientWidth / img.naturalWidth,
        y: img.clientHeight / img.naturalHeight
      });
    };

    updateScale();
    const observer = new ResizeObserver(updateScale);
    observer.observe(img);

    return () => observer.disconnect();
  }, [selected?.filePath, selected?.annotations]);

  async function reInferImage() {
    if (!selected) return;
    setLoading(true);

    const res = await fetch(`${BASE_URL}/infer/model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_path: selected.filePath }),
    });

    const data = await res.json();
    setLoading(false);

    const annotations = (data.annotations || []).map((a: any, i: number) => ({
      ...a,
      id: i + 1,
      active: true,
    }));

    setSelected({
      filePath: selected.filePath,
      fileName: selected.fileName,
      annotations,
    });
    

    const img = imgRef.current;
    if (img && img.complete) {
      setScale({
        x: img.clientWidth / img.naturalWidth,
        y: img.clientHeight / img.naturalHeight,
      });
    } else if (img) {
      img.onload = () => {
        setScale({
          x: img.clientWidth / img.naturalWidth,
          y: img.clientHeight / img.naturalHeight,
        });
      };
    }

    console.log("Box đã được cập nhật lại từ model");
  }


  useEffect(() => {
    fetchFiles();
  }, []);

  return (
    <div className="flex h-screen bg-white text-gray-900 font-sans">
      {/* LEFT: IMAGE LIST */}
      <div className="w-1/5 border-r border-gray-300 p-4 overflow-y-auto">
        <h2 className="font-semibold mb-3 text-lg text-gray-700 uppercase tracking-wide">
          Danh sách ảnh
        </h2>
        {files.map((f, i) => {
          const isActive = currentIndex === i;
          const hasBox = f.hasBox;

          // Chọn màu rõ ràng hơn
          let bgClass = "";
          if (isActive) bgClass = "bg-blue-100 border border-blue-300";
          else if (hasBox)
            bgClass = "bg-orange-100 hover:bg-orange-200 border border-orange-300";
          else
            bgClass = "bg-green-50 hover:bg-green-100 border border-green-300";

          return (
            <div
              key={f.fileName}
              onClick={() => {
                setCurrentIndex(i);
                inferImage(f.filePath);
              }}
              className={`cursor-pointer p-2 mb-1 rounded transition ${bgClass}`}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium">{f.fileName}</span>
                {isActive ? (
                  <span className="text-xs text-blue-600 font-semibold">🔵</span>
                ) : hasBox ? (
                  <span className="text-xs text-orange-600 font-semibold">DB</span>
                ) : (
                  <span className="text-xs text-green-600 font-semibold">NEW</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* CENTER: IMAGE PREVIEW */}
      <div className="w-3/5 flex flex-col items-center justify-center relative border-x border-gray-300">
        {loading && (
          <div className="absolute top-1/2 text-gray-500">Đang chạy mô hình...</div>
        )}

        {!loading && selected ? (
          <>
            <div className="relative">
            <img
              key={selected.fileName + selected.annotations.length}   // ✅ thêm key động
              ref={imgRef}
              src={selected.filePath}   // chỉ giữ path tương đối thôi
              alt={selected.fileName}
              className="rounded max-h-[75vh] border border-gray-200 shadow-sm"
            />
              {/* HIỂN THỊ BOX */}
              {selected.annotations?.map((a: any) => {
                if (!a.active) return null;

                // Kiểm tra bbox hợp lệ
                if (!Array.isArray(a.bbox) || a.bbox.length < 4) {
                  console.warn("⚠️ Invalid bbox:", a);
                  return null;
                }

                // Ép kiểu float và kiểm tra hợp lệ
                const [x1, y1, x2, y2] = a.bbox.map(Number);
                const left = x1 * scale.x;
                const top = y1 * scale.y;
                const width = (x2 - x1) * scale.x;
                const height = (y2 - y1) * scale.y;
                console.log("BOX:", { file: selected.fileName, x1, y1, x2, y2, left, top, width, height, scale });

                // Nếu có giá trị NaN, bỏ qua box đó để tránh crash
                if (![left, top, width, height].every((v) => Number.isFinite(v))) {
                  console.warn("⚠️ Skip invalid box:", { x1, y1, x2, y2, scale });
                  return null;
                }

                return (
                  <div
                    key={a.id}
                    className="absolute rounded-sm border-2 border-green-500"
                    style={{
                      left,
                      top,
                      width,
                      height,
                      backgroundColor: "rgba(34,197,94,0.15)",
                      boxShadow: "0 0 4px rgba(0,128,0,0.4)",
                    }}
                  >
                    <div className="absolute -top-5 left-0 bg-black/75 text-white text-[11px] font-semibold px-2 py-[1px] rounded-t">
                      #{a.id} {a.label} ({(a.confidence * 100).toFixed(1)}%)
                    </div>
                  </div>
                );
              })}

            </div>

            {/* BUTTONS */}
            <div className="flex gap-4 mt-5">
              <button
                onClick={prevImage}
                disabled={currentIndex === 0}
                className="px-5 py-2 bg-gray-200 hover:bg-gray-300 rounded disabled:opacity-50"
              >
                Ảnh trước
              </button>

              <button
                onClick={saveImage}
                className="px-5 py-2 bg-green-500 hover:bg-green-600 text-white font-semibold rounded"
              >
                Lưu thay đổi
              </button>

              <button
                onClick={reInferImage}
                className="px-5 py-2 bg-yellow-400 hover:bg-yellow-500 text-black font-semibold rounded"
              >
                Dùng lại mô hình
              </button>

              <button
                onClick={nextImage}
                disabled={currentIndex >= files.length - 1}
                className="px-5 py-2 bg-gray-200 hover:bg-gray-300 rounded disabled:opacity-50"
              >
                Ảnh sau
              </button>
            </div>

          </>
        ) : (
          <div className="text-gray-400">Chọn một ảnh bên trái để xem</div>
        )}
      </div>

      {/* RIGHT: LABEL INFO */}
      <div className="w-1/5 border-l border-gray-300 p-4 overflow-y-auto">
        <h2 className="font-semibold mb-3 text-lg text-gray-700 uppercase tracking-wide">
          Thông tin label
        </h2>
        {selected?.annotations?.length ? (
          selected.annotations.map((a: any, i: number) => (
            <div
              key={a.id}
              className={`mb-3 border-b border-gray-200 pb-2 ${
                a.active ? "" : "opacity-50 line-through"
              }`}
            >
              <div className="flex justify-between items-center mb-2">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={a.active}
                    onChange={() => toggleActive(i)}
                  />
                  <span className="font-medium text-gray-700">#{a.id}</span>
                </label>
                <span className="text-sm text-gray-500">
                  {(a.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <input
                className="w-full border border-gray-300 px-2 py-1 rounded text-sm focus:ring-1 focus:ring-blue-400 outline-none"
                value={a.label}
                onChange={(e) => updateLabel(i, e.target.value)}
              />
            </div>
          ))
        ) : (
          <div className="text-gray-400">Không có label nào.</div>
        )}
      </div>
    </div>
  )
}