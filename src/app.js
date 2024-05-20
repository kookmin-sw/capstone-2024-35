require('dotenv').config();

const express = require('express');
const expressLayouts = require('express-ejs-layouts');
const methodOverride = require('method-override');
const flash = require('connect-flash');
const connectDB = require('./config/db');
const session = require('express-session');
const passport = require('passport');
const MongoStore = require('connect-mongo');
const morgan = require('morgan');
import('ejs-lint').then(ejsLint => {

}).catch(err => {
  console.error('모듈을 불러오는 동안 오류가 발생했습니다.', err)
});

const app = express();
const port = 5000 || process.env.PORT;

app.use(session({
  secret: 'keyboard cat',
  resave: false,
  saveUninitialized: true,
  store: MongoStore.create({
    mongoUrl: process.env.MONGODB_URI
  }),
  cookie: {
    maxAge: 1000 * 60 * 60 * 24 * 7, //1주
  }
  //cookie: { maxAge: new Date ( Date.now() + (3600000) ) } 
  // Date.now() - 30 * 24 * 60 * 60 * 1000
}));


//모건
app.use(morgan('dev'))

app.use(flash({ sessionKeyName: 'flashMessage' }));

app.use(passport.initialize());
app.use(passport.session());


app.use(express.urlencoded({extended: true}));
app.use(express.json());
app.use(methodOverride("_method"));
// 데이터베이스 연결
connectDB();  

// 정적 파일들
app.use(express.static('public'));

// 템플릿 엔진
app.use(expressLayouts);
app.set('layout', './layouts/main');
app.set('view engine', 'ejs');



// 라우팅
app.use('/', require('./routes/auth'));
app.use('/', require('./routes/index'));
app.use('/', require('./routes/dashboard'));
app.use('/', require('./routes/employee'));
app.use('/', require('./routes/worksite'));


// 404 설정
app.get('*', function(req, res) {
  //res.status(404).send('404 Page Not Found.')
  res.status(404).render('404');
})


app.listen(port, () => {
  console.log(`App listening on port ${port}`);
});