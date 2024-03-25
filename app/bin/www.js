"use strict";

const bin = require("../app");

const PORT = 3000;

bin.listen(PORT, () => {
    console.log("서버 가동")
});