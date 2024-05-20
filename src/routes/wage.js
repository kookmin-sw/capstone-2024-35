const express = require('express');
const router = express.Router();
const { isLoggedIn } = require('../middleware/checkAuth');
const wageController = require('../controllers/wageController');

router.get('/wage', isLoggedIn, wageController.wage);

module.exports = router;