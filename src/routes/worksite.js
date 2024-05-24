const express = require('express');
const router = express.Router();
const { isLoggedIn } = require('../middleware/checkAuth');
const worksiteController = require('../controllers/worksiteController');
const smsController = require('../controllers/smsController')

router.get('/python', smsController.parseMessage);
router.get('/sms', smsController.getsms);
router.post('/sms', smsController.parseMessage);
router.get('/worksite', isLoggedIn, worksiteController.worksite);
router.get('/worksite/add', isLoggedIn, worksiteController.addWorksite);
router.post('/worksite/post', isLoggedIn, worksiteController.postWorksite);
router.post('/worksite/search', isLoggedIn, worksiteController.searchWorksite);
router.get('/worksite/:id', isLoggedIn, worksiteController.showWorksite);
router.get('/worksite/:id/payment', isLoggedIn, worksiteController.showWorksitePayment);
router.get('/worksite/:id/hire', isLoggedIn, worksiteController.matchToWorksite);
router.post('/worksite/:id/hire/:eid', isLoggedIn, worksiteController.worksiteHireEmployee);
router.get('/worksite/:id/edit', isLoggedIn, worksiteController.editWorksite);
router.put('/worksite/:id/edit', isLoggedIn, worksiteController.putWorksite);
router.delete('/worksite/:id/delete', isLoggedIn, worksiteController.deleteWorksite);
router.delete('/worksite/:id/delete/:eid', isLoggedIn, worksiteController.deleteMatchedEmployee);


module.exports = router;