const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { exec } = require('child_process');
const path = require('path');
const yahooFinance = require('yahoo-finance2').default;

const app = express();
const PORT = 5000;
const cors = require('cors');

app.use(cors({
  origin: 'https://stockwisely.netlify.app/',
  credentials: true,
}));
// MongoDB Connection
mongoose.connect(process.env.MONGO_URI);


// User Schema for Authentication and Watchlist
const userSchema = new mongoose.Schema({
  username: String,
  email: { type: String, unique: true },
  password: String,
  watchlist: [
    {
      ticker: String,
      companyName: String,
      prices: [Number], // Array of stock prices for the ticker
      dates: [String], // Array of dates for the stock data
    },
  ],
});

const User = mongoose.model('User', userSchema);

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use('/graphs', express.static(path.join(__dirname, 'graphs')));

// Routes

// Signup Route
app.post('/signup', async (req, res) => {
  const { email, password, username } = req.body;

  if (!email || !password || !username) {
    return res.status(400).json({ message: 'All fields are required' });
  }

  try {
    const userExists = await User.findOne({ email });
    if (userExists) {
      return res.status(400).json({ message: 'Email already exists' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ email, password: hashedPassword, username });
    await newUser.save();
    res.status(201).json({ message: 'User created successfully!' });
  } catch (error) {
    console.error('Error creating user:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Login Route
app.post('/login', async (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ message: 'Email and password are required' });
  }

  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);

    if (!isPasswordValid) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    const token = jwt.sign({ id: user._id }, 'secretkey', { expiresIn: '1h' });
    res.json({ token });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Update Password Route
app.post('/api/user/update-password', async (req, res) => {
  const { email, password } = req.body;
  const token = req.headers.authorization && req.headers.authorization.split(' ')[1];

  if (!token) {
    return res.status(401).json({ message: 'No token provided' });
  }

  try {
    // Verify the token
    const decoded = jwt.verify(token, 'secretkey');
    const user = await User.findOne({ email });

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Hash the new password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Update the user's password
    user.password = hashedPassword;
    await user.save();

    res.status(200).json({ message: 'Password updated successfully' });
  } catch (error) {
    console.error('Error updating password:', error);
    res.status(500).json({ message: 'Server error' });
  }
});



// Fetch Stock Data from Yahoo Finance
app.get('/api/stocks/:ticker', async (req, res) => {
  const { ticker } = req.params;

  try {
    const quote = await yahooFinance.quote(ticker);
    const history = await yahooFinance.historical(ticker, {
      period1: '2021-01-01',
      interval: '1d',
    });

    const stockData = {
      companyName: quote.shortName || 'Unknown',
      currentPrice: quote.regularMarketPrice,  // Add current price here
      prices: history.map(item => item.close),
      dates: history.map(item => item.date.toISOString().split('T')[0]),
    };

    res.json(stockData);
  } catch (error) {
    console.error('Error fetching stock data from Yahoo Finance:', error);
    res.status(500).json({ message: 'Error fetching stock data' });
  }
});




// Prediction Route (using a Python script or model)
app.post('/predict', (req, res) => {
  const { ticker, predictionDate } = req.body;

  if (!ticker || !predictionDate) {
    return res.status(400).json({ error: 'Ticker and prediction date are required.' });
  }

  const scriptPath = path.join(__dirname, 'model', 'model.py');
  const command = `python "${scriptPath}" "${ticker}" "${predictionDate}"`;

  exec(command, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing script: ${stderr}`);
      return res.status(500).json({ error: `Failed to process prediction: ${stderr}` });
    }

    try {
      const response = JSON.parse(stdout);
      res.json(response);
    } catch (parseError) {
      console.error('Error parsing Python script output:', parseError);
      res.status(500).json({ error: 'Unexpected script output.' });
    }
  });
});

// Watchlist Routes (Add, Remove, Fetch)
app.post('/watchlist/add', async (req, res) => {
  const { email, ticker, companyName = 'Unknown', prices = [], dates = [] } = req.body;

  if (!email || !ticker) {
    return res.status(400).json({ message: 'Email and ticker are required' });
  }

  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    const stockExists = user.watchlist.some(stock => stock.ticker === ticker);
    if (stockExists) {
      return res.status(400).json({ message: 'Stock already in watchlist' });
    }

    user.watchlist.push({ ticker, companyName, prices, dates });
    await user.save();

    res.status(200).json({ message: 'Stock added to watchlist' });
  } catch (error) {
    console.error('Error adding stock to watchlist:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

app.post('/watchlist/remove', async (req, res) => {
  const { email, ticker } = req.body;

  if (!email || !ticker) {
    return res.status(400).json({ message: 'Email and ticker are required' });
  }

  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    const updatedWatchlist = user.watchlist.filter(stock => stock.ticker !== ticker);
    user.watchlist = updatedWatchlist;
    await user.save();

    res.status(200).json({ message: 'Stock removed from watchlist' });
  } catch (error) {
    console.error('Error removing stock from watchlist:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Fetch Watchlist Route
app.get('/watchlist/:email', async (req, res) => {
  const { email } = req.params;

  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    res.status(200).json({ watchlist: user.watchlist });
  } catch (error) {
    console.error('Error fetching watchlist:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Start the Server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
