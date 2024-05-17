const mongoose = require('mongoose');

const Schema = mongoose.Schema;
const careerSchema = new Schema({
	employee: {
    	type: Schema.Types.ObjectId,
        ref: 'Employee'
    },
    worksite: {
    	type: Schema.Types.ObjectId,
        ref: 'Worksite'
    },

});

module.exports = mongoose.model('Career', careerSchema)