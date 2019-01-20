import * as R from 'ramda';

import defaultImage from '../image.jpg';

const setLoading = (button) => {
  button.classList.add('is-loading');
}

const setDone = (button) => {
  button.classList.remove('is-loading');
}

let model = null;

const enableCopyAndDownload = () => {
  model.copyButton.disabled = false;
  model.downloadButton.disabled = false;
}

const writeSourceInfo = () => {
  const { canvases: { source }, sourceInfo } = model
  sourceInfo.innerText = `(${source.width}x${source.height})`
};

const writeTargetInfo = () => {
  const { canvases: { target }, targetInfo } = model
  targetInfo.innerText = `(${target.width}x${target.height})`
};

const generateKernelFromFunction = (kernelSize, f) => {
  return R.range(0, kernelSize).map(y => R.range(0, kernelSize).map(x => f(x, y)));
};

const generateZeroKernel = (kernelSize) => {
  return generateKernelFromFunction(kernelSize, () => 0);
};

const allKernels = [
  'averagingKernel',
  'gaussianKernel',
  'derivativeKernel',
  'prewittKernel',
  'sobelKernel',
  'customKernel',
];

const generateKernel = () => {
  const { kernelType, kernelSize, sigma, isVertical, customKernel } = model;
  if (kernelType === 'customKernel') return customKernel;

  const middle =  Math.floor(kernelSize / 2);
  const f = {
    averagingKernel: (_x, _y) => 1 / kernelSize**2,
    gaussianKernel: (x, y) => {
      const dx = x - middle, dy = middle - y;
      return Math.exp(-(dx**2 + dy**2)/(2*sigma**2));
    },
    derivativeKernel: (x, y) => {
      let [a, b] = isVertical ? [x, y] : [y, x];
      return (a !== middle) ?
        0 :
        (b === middle - 1) ? -1 :
        (b === middle + 1) ? 1 :
        0;
    },
    prewittKernel: (x, y) => (isVertical ? middle - y : x - middle),
    sobelKernel: (x, y) => {
      const dx = x - middle, dy = middle - y, scale = dy**2 + dx**2;
      return scale === 0 ? 0 : (isVertical ? dy : dx) / scale;
    }
  }[kernelType];

  if (f === undefined) throw new Error('Kernel not found: ' + kernelType);

  const result = generateKernelFromFunction(kernelSize, f);

  return kernelType === 'gaussianKernel' ? normalizeKernel(result) : result;
};

const writeKernelToDom = () => {
  const { kernel, kernelSize } = model;
  const table = document.getElementById('kernelTable');
  const trs = R.range(0, kernelSize).map(y => {
    const tr = document.createElement('tr');
    R.range(0, kernelSize).forEach(x => {
      const td = document.createElement('td');
      const input = document.createElement('input');
      input.setAttribute('class', 'input kernel');
      input.setAttribute('type', 'text');
      input.setAttribute('value', kernel[y][x].toString());
      input.addEventListener('input', () => {
        const value = parseFloat(input.value);
        if (isNaN(value) || kernel[y][x] === value) return;
        model.customKernel = kernel;
        model.customKernel[y][x] = value;
        model.kernelType = 'customKernel';
        model.zeroButton.disabled = false;
        document.getElementById('customKernel').checked = true;
      });
      td.appendChild(input);
      tr.appendChild(td);
    });
    return tr;
  });
  while (table.firstChild) {
    table.removeChild(table.firstChild);
  }
  trs.forEach(tr => {
    table.appendChild(tr);
  })
};

const normalizeKernel = (kernel) => {
  const sum = R.sum(kernel.map(R.sum));
  return sum === 0 ? kernel : kernel.map(row => row.map(x => x / sum));
};

const applyKernel = (kernel, kernelSize) => {
  const { wasm , memory } = model;
  const flatKernel = R.unnest(kernel);

  const kernelHandle = wasm.Kernel.new(kernelSize);
  const contents = new Float32Array(memory.buffer, kernelHandle.content(), kernelSize**2);
  for (let i = 0; i < kernelSize**2; i++) {
    contents[i] = flatKernel[i];
  }
  return kernelHandle;
};

const runConvolution = () => {
  const {
    wasm,
    kernel,
    kernelSize,
    canvases: { source, target },
    convolveButton,
  } = model;

  setLoading(convolveButton);
  const start = performance.now();

  let kernelHandle = applyKernel(kernel, kernelSize)
  wasm.run_convolution(source, target, kernelHandle);

  console.log(performance.now() - start);
  writeTargetInfo();
  setDone(convolveButton);
  enableCopyAndDownload();
};

const runMedianFilter = () => {
  const { wasm, canvases: { source, target } } = model;

  const start = performance.now();
  wasm.run_median_filter(source, target, model.kernelSize);
  console.log(performance.now() - start);
  writeTargetInfo();
  enableCopyAndDownload();
};

const runBilateralFilter = () => {
  const { wasm, canvases: { source, target } } = model;

  const start = performance.now();
  wasm.run_bilateral_filter(source, target, model.kernelSize, model.sigma);
  console.log(performance.now() - start);
  writeTargetInfo();
  enableCopyAndDownload();
};

const runGammaCorrection = () => {
  const { wasm, canvases: { source, target } } = model;

  const start = performance.now();
  wasm.run_gamma_correction(source, target, model.gamma);
  console.log(performance.now() - start);
  writeTargetInfo();
  enableCopyAndDownload();
};

const generateDitherPattern = () => {
  return [
    0, 8, 2, 10,
    12, 4, 14, 6,
    3, 11, 1, 9,
    15, 7, 13, 5
  ];
};

const runLumaConversion = () => {
  const { wasm, canvases: { source, target } } = model;

  const start = performance.now();
  wasm.run_luma_conversion(source, target);
  console.log(performance.now() - start);
  writeTargetInfo();
  enableCopyAndDownload();
};

const runHalftoneConversion = () => {
  const { wasm, canvases: { source, target } } = model;

  const start = performance.now();
  wasm.run_density_pattern_halftone(source, target);
  console.log(performance.now() - start);
  writeTargetInfo();
  enableCopyAndDownload();
};

const runDitherHalftone = () => {
  const { wasm, canvases: { source, target } } = model;

  const start = performance.now();
  wasm.run_dither_halftone(source, target, model.ditherPattern);
  console.log(performance.now() - start);
  writeTargetInfo();
  enableCopyAndDownload();
};

const drawImage = (image) => {
  const { source, sourceCtx } = model.canvases;
  source.width = image.naturalWidth;
  source.height = image.naturalHeight;
  writeSourceInfo();
  sourceCtx.drawImage(image, 0, 0);
};

const handleImageUpload = (e) => {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      drawImage(img);
      model.defaultImage = img;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(e.target.files[0]);
  document.getElementById('file-name').innerText = e.target.files[0].name;
};

const drawDefaultImage = () => {
  model.convolveButton.disabled = true;
  const img = new Image();
  img.onload = () => {
    drawImage(img);
    model.convolveButton.disabled = false;
    model.defaultImage = img;
  };
  img.src = defaultImage;
};

const listenForCanvasClick = (canvas) => {
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect()
    const x = Math.round((e.clientX - rect.left) * canvas.width / rect.width);
    const y = Math.round((e.clientY - rect.top) * canvas.height / rect.height);
    console.log(`(${x}, ${y})`);
    model.pointList.push({ x, y });
    while (model.pointList.length > 4) {
      model.pointList.shift();
    }
    console.log(model.pointList);
  });
};

const main = (wasm, memory) => {
  model = {
    wasm,
    memory,
    kernel: null,
    customKernel: null,
    kmeans: 10,
    kernelSize: 3,
    kernelType: 'sobelKernel',
    sigma: null,
    isVertical: false,
    ditherPattern: generateDitherPattern(),
    gamma: 2.2,
    pointList: [],
    canvases: {
      source: document.getElementById('sourceCanvas'),
      target: document.getElementById('targetCanvas'),
      sourceCtx: null,
      targetCtx: null,
    },
    defaultImage: null,
    sourceInfo: document.getElementById('sourceInfo'),
    targetInfo: document.getElementById('targetInfo'),
    convolveButton: document.getElementById('convolveButton'),
    copyButton: document.getElementById('copyButton'),
    downloadButton: document.getElementById('downloadButton'),
    zeroButton: document.getElementById('zeroButton'),
  };

  model.kernel = generateKernel();
  model.customKernel = generateKernelFromFunction(model.kernelSize, () => 0);
  model.canvases.sourceCtx = model.canvases.source.getContext('2d');
  model.canvases.targetCtx = model.canvases.target.getContext('2d');
  model.sigma = model.kernelSize / 2;

  drawDefaultImage();
  writeKernelToDom();
  listenForCanvasClick(model.canvases.source);

  allKernels.forEach((kernelName) => {
    const radioButton = document.getElementById(kernelName);
    if (model.kernelType === kernelName) {
      radioButton.checked = true;
      model.zeroButton.disabled = model.kernelType !== 'customKernel';
    }
    radioButton.addEventListener('input', () => {
      if (radioButton.checked && model.kernelType !== kernelName) {
        model.kernelType = kernelName;
        model.kernel = generateKernel();
        model.zeroButton.disabled = model.kernelType !== 'customKernel';
        writeKernelToDom();
      }
    });
  });

  const imageLoader = document.getElementById('imageLoader');
  imageLoader.addEventListener('change', handleImageUpload, false);

  const kernelSizeInput = document.getElementById('kernelSizeInput');
  kernelSizeInput.value = model.kernelSize;
  kernelSizeInput.addEventListener('input', () => {
    const value = parseInt(kernelSizeInput.value);
    if (isNaN(value) || value % 2 === 0 || model.kernelSize === value) return;
    model.kernelSize = value;
    model.customKernel = generateZeroKernel(model.kernelSize);
    model.kernel = generateKernel();
    writeKernelToDom();
  });

  model.convolveButton.addEventListener('click', () => {
    setTimeout(runConvolution, 0);
  });

  model.copyButton.addEventListener('click', () => {
    model.canvases.source.width =  model.canvases.target.width;
    model.canvases.source.height = model.canvases.target.height;
    writeSourceInfo();
    model.canvases.sourceCtx.drawImage(model.canvases.target, 0, 0);
  });

  // chrome workaround: c.f. https://stackoverflow.com/a/37151835
  model.downloadButton.addEventListener('click', () => {
    setLoading(model.downloadButton);
    model.canvases.target.toBlob(blob => {
      const a = document.createElement('a');
      a.download = "result.png";
      a.href = URL.createObjectURL(blob);
      a.hidden = true;
      a.onclick = () => {
        requestAnimationFrame(() => {
          URL.revokeObjectURL(a.href);
        });
      };

      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setDone(model.downloadButton);
    });
  });

  const projectButton = document.getElementById('projectButton');
  projectButton.addEventListener('click', () => {
    setTimeout(() => {
      const x = model.pointList.map(({x}) => x);
      const y = model.pointList.map(({y}) => y);
      const w = model.canvases.source.width;
      const h = model.canvases.source.height;
      const nx = [0, w, 0, w];
      const ny = [0, 0, h, h];
      if (model.pointList.length === 4) {
        const start = performance.now();
        model.wasm.run_projection(model.canvases.source, model.canvases.target, nx, ny, x, y);
        console.log(performance.now() - start);
        writeTargetInfo();
        enableCopyAndDownload();
      }
    }, 0);
  });

  const kmeansInput = document.getElementById('kmeansInput');
  kmeansInput.value = model.kmeans;
  kmeansInput.addEventListener('input', () => {
    const value = parseInt(kmeansInput.value);
    if (isNaN(value) || model.kmeans === value) return;
    model.kmeans = value;
  });

  const kmeansButton = document.getElementById('kmeansButton');
  kmeansButton.addEventListener('click', () => {
    let k = model.kmeans;
    setTimeout(() => {
        const start = performance.now();
        model.wasm.run_image_segmentation(model.canvases.source, model.canvases.target, k);
        console.log(performance.now() - start);
        writeTargetInfo();
        enableCopyAndDownload();
    }, 0);
  });


  const histogramEqualizationButton = document.getElementById('histogramEqualizationButton');
  histogramEqualizationButton.addEventListener('click', () => {
    setTimeout(() => {
        const start = performance.now();
        model.wasm.run_histogram_equalization(model.canvases.source, model.canvases.target);
        console.log(performance.now() - start);
        writeTargetInfo();
        enableCopyAndDownload();
    }, 0);
  });


  const lumaButton = document.getElementById('lumaButton');
  lumaButton.addEventListener('click', () => {
    setTimeout(runLumaConversion, 0);
  });

  const halftoneButton = document.getElementById('halftoneButton');
  halftoneButton.addEventListener('click', () => {
    setTimeout(runHalftoneConversion, 0);
  });

  const ditherButton = document.getElementById('ditherButton');
  ditherButton.addEventListener('click', () => {
    setTimeout(runDitherHalftone, 0);
  });

  const resetButton = document.getElementById('resetButton');
  resetButton.addEventListener('click', () => {
    drawImage(model.defaultImage);
  });

  const normalizeButton = document.getElementById('normalizeButton');
  normalizeButton.addEventListener('click', () => {
    model.kernel = normalizeKernel(model.kernel);
    writeKernelToDom();
  });

  model.zeroButton.addEventListener('click', () => {
    model.kernel = model.customKernel = generateZeroKernel(model.kernelSize);
    writeKernelToDom();
  });

  const isVerticalCheckbox = document.getElementById('isVerticalCheckbox');
  isVerticalCheckbox.checked = model.isVertical;
  isVerticalCheckbox.addEventListener('change', () => {
    if (model.isVertical !== isVerticalCheckbox.checked &&
        [ 'prewittKernel', 'sobelKernel'].includes(model.kernelType)) {
      model.isVertical = isVerticalCheckbox.checked;
      model.kernel = generateKernel();
      writeKernelToDom();
    }
  });

  const sigmaInput = document.getElementById('sigmaInput');
  sigmaInput.value = model.sigma;
  sigmaInput.addEventListener('input', () => {
    const value = parseFloat(sigmaInput.value);
    if (isNaN(value) || model.sigma === value) return;
    model.sigma = value;
    model.kernel = generateKernel();
    writeKernelToDom();
  });

  const medianButton = document.getElementById('medianButton');
  medianButton.addEventListener('click', () => {
    setTimeout(runMedianFilter, 0);
  });

  const bilateralButton = document.getElementById('bilateralButton');
  bilateralButton.addEventListener('click', () => {
    setTimeout(runBilateralFilter, 0);
  });

  const gammaInput = document.getElementById('gammaInput');
  gammaInput.value = model.gamma;
  gammaInput.addEventListener('input', () => {
    const value = parseFloat(gammaInput.value);
    if (isNaN(value) || model.gamma === value) return;
    model.gamma = value;
  });

  const gammaButton = document.getElementById('gammaButton');
  gammaButton.addEventListener('click', () => {
    setTimeout(runGammaCorrection, 0);
  });
};

Promise.all([
  import("../crate/pkg/simple_image_editor"),
  import("../crate/pkg/simple_image_editor_bg")
]).then(([module, { memory }]) => {
  module.init();
  main(module, memory);
}).catch(err => {
  console.error(err);
})
