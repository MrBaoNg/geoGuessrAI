const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });
app.use(cors());

function clearUploadsSync() {
  const dir = 'uploads/';
  if (!fs.existsSync(dir)) return;
  for (const file of fs.readdirSync(dir)) {
    try {
      fs.unlinkSync(path.join(dir, file));
    } catch (err) {
      console.warn(`Could not delete ${file}:`, err);
    }
  }
}

app.post('/upload', upload.single('image'), (req, res) => {
  const imagePath = path.resolve(req.file.path);
  
  const py = spawn('conda', [
    'run', '-n', 'geoenv',  // or just 'python3' if you’re on a straight pip install
    'python', 'geo_predict.py', 
    imagePath
  ], { stdio: ['ignore','pipe','pipe'] });

  let stdout = '';
  py.stdout.on('data', chunk => { stdout += chunk.toString() });
  py.stderr.on('data', chunk => { console.error('PY ERR:', chunk.toString()) });

  py.on('close', code => {
    clearUploadsSync();
    if (code !== 0) {
      return res.status(500).json({ error: 'Model failed', code });
    }

    res.json({ message: stdout.trim() });
  });
});

app.listen(5000, () => {
  console.log('Backend running on port 5000');
});
