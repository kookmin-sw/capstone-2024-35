const express = require('express');
const router = express.Router();
const mainController = require('../controllers/mainController');
const { isLoggedIn } = require('../middleware/checkAuth');

/**
 * App Routes 
*/
router.get('/', mainController.homepage);
router.get('/about', isLoggedIn, mainController.about);




module.exports = router;