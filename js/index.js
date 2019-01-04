import * as R from 'ramda';

import defaultImage from '../image.jpg';

let model = null;

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
    copyButton,
    downloadButton,
  } = model;

  convolveButton.disabled = true;
  const start = performance.now();

  let kernelHandle = applyKernel(kernel, kernelSize)
  wasm.run_convolution(source, target, kernelHandle);

  console.log(performance.now() - start);
  convolveButton.disabled = false;
  copyButton.disabled = false;
  downloadButton.disabled = false;
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
  const { wasm, canvases: { source, target }, copyButton, downloadButton } = model;

  const start = performance.now();
  wasm.run_luma_conversion(source, target);
  console.log(performance.now() - start);
  copyButton.disabled = false;
  downloadButton.disabled = false;
};

const runHalftoneConversion = () => {
  const { wasm, canvases: { source, target }, copyButton, downloadButton  } = model;

  const start = performance.now();
  wasm.run_density_pattern_halftone(source, target);
  console.log(performance.now() - start);
  copyButton.disabled = false;
  downloadButton.disabled = false;
};

const runDitherHalftone = () => {
  const { wasm, canvases: { source, target }, copyButton, downloadButton  } = model;

  const start = performance.now();
  wasm.run_dither_halftone(source, target, model.ditherPattern);
  console.log(performance.now() - start);
  copyButton.disabled = false;
  downloadButton.disabled = false;
};

const drawImage = (image) => {
  const { source, sourceCtx } = model.canvases;
  source.width = image.naturalWidth;
  source.height = image.naturalHeight;
  sourceCtx.drawImage(image, 0, 0);
};

const handleImageUpload = (e) => {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      drawImage(img);
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
  };
  img.src = defaultImage;
};

const main = (wasm, memory) => {
  model = {
    wasm,
    memory,
    kernel: null,
    customKernel: null,
    kernelSize: 3,
    kernelType: 'sobelKernel',
    sigma: null,
    isVertical: false,
    ditherPattern: generateDitherPattern(),
    canvases: {
      source: document.getElementById('sourceCanvas'),
      target: document.getElementById('targetCanvas'),
      sourceCtx: null,
      targetCtx: null,
    },
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
    model.canvases.sourceCtx.drawImage(model.canvases.target, 0, 0);
  });

  // chrome workaround: c.f. https://stackoverflow.com/a/37151835
  model.downloadButton.addEventListener('click', () => {
    model.downloadButton.disabled = true;
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
      model.downloadButton.disabled = false;
    });
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
